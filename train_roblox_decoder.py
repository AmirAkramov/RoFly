import numpy as np
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

CONTROLLED_DATA_FILE = Path("roblox_controlled_demonstrations.npz")
LEGACY_DATA_FILE = Path("roblox_demonstrations.npz")
DATA_FILE = (
    CONTROLLED_DATA_FILE
    if CONTROLLED_DATA_FILE.exists()
    else LEGACY_DATA_FILE
)
OUTPUT_FILE = Path("roblox_decoder.npz")

PCA_COMPONENTS = 200
TEST_RATIO = 0.20

EPOCHS = 1200
LEARNING_RATE = 0.035
L2 = 0.0005

RANDOM_SEED = 42

# ============================================================
# LABELS
# ============================================================

LABELS = [
    "W",
    "A",
    "S",
    "D",
    "CAM_LEFT",
    "CAM_RIGHT",
    "W_CAM_LEFT",
    "W_CAM_RIGHT",
]

# ============================================================
# DECODER TARGETS
# ============================================================

FORWARD_LABELS = {
    "W": 1,
    "W_CAM_LEFT": 1,
    "W_CAM_RIGHT": 1,

    "S": 2,

    "A": 0,
    "D": 0,
    "CAM_LEFT": 0,
    "CAM_RIGHT": 0,
}

TURN_LABELS = {
    "A": 1,
    "W_CAM_LEFT": 1,

    "D": 2,
    "W_CAM_RIGHT": 2,

    "W": 0,
    "S": 0,
    "CAM_LEFT": 0,
    "CAM_RIGHT": 0,
}

CAMERA_LABELS = {
    "CAM_LEFT": 1,
    "W_CAM_LEFT": 1,

    "CAM_RIGHT": 2,
    "W_CAM_RIGHT": 2,

    "W": 0,
    "A": 0,
    "S": 0,
    "D": 0,
}

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ROBLOX NEURAL DECODER TRAINING")
print("BALANCED + STRATIFIED + PCA")
print("=" * 70)
print()

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_FILE}. "
        "Make sure the script is in your doomfly folder."
    )

data = np.load(DATA_FILE)

X = np.asarray(data["features"], dtype=np.float32)
labels = np.asarray(data["labels"]).astype(str)

print(f"Loaded demonstrations: {len(X)}")
print(f"Neural feature size:   {X.shape[1]}")
print()

# ============================================================
# SHOW ORIGINAL DATA BALANCE
# ============================================================

print("Original dataset:")
for label in LABELS:
    count = np.sum(labels == label)
    print(f"  {label:15s}: {count}")

print()

# ============================================================
# STRATIFIED SPLIT
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

train_indices = []
test_indices = []

for label in LABELS:

    indices = np.where(labels == label)[0]

    if len(indices) == 0:
        continue

    indices = indices.copy()
    rng.shuffle(indices)

    if len(indices) == 1:
        test_count = 0
    else:
        test_count = max(1, int(round(len(indices) * TEST_RATIO)))

        # Never put all samples into test
        test_count = min(test_count, len(indices) - 1)

    test_indices.extend(indices[:test_count])
    train_indices.extend(indices[test_count:])


train_indices = np.asarray(train_indices, dtype=np.int64)
test_indices = np.asarray(test_indices, dtype=np.int64)

rng.shuffle(train_indices)
rng.shuffle(test_indices)

X_train = X[train_indices]
X_test = X[test_indices]

y_train_labels = labels[train_indices]
y_test_labels = labels[test_indices]

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")
print()

# ============================================================
# NORMALIZATION
# ============================================================

print("Normalizing neural features...")

mean = X_train.mean(axis=0)

std = X_train.std(axis=0)
std[std < 1e-6] = 1.0

X_train_n = (X_train - mean) / std
X_test_n = (X_test - mean) / std

# ============================================================
# PCA
# ============================================================

print()
print("Running PCA...")

# Center again before PCA.
pca_mean = X_train_n.mean(axis=0)

X_train_n -= pca_mean
X_test_n -= pca_mean

n_samples = X_train_n.shape[0]

# Gram matrix is much smaller than the full covariance matrix.
gram = (X_train_n @ X_train_n.T) / max(1, n_samples - 1)

eigvals, eigvecs = np.linalg.eigh(gram)

order = np.argsort(eigvals)[::-1]

eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

# Remove numerical noise.
valid = eigvals > 1e-8

eigvals = eigvals[valid]
eigvecs = eigvecs[:, valid]

k = min(PCA_COMPONENTS, len(eigvals))

eigvals = eigvals[:k]
eigvecs = eigvecs[:, :k]

# Convert Gram eigenvectors into feature-space PCA directions.
components = (
    X_train_n.T @ eigvecs
) / np.sqrt(
    np.maximum(eigvals * max(1, n_samples - 1), 1e-8)
)

# Normalize components.
component_norms = np.linalg.norm(components, axis=0)
component_norms[component_norms < 1e-8] = 1.0

components /= component_norms

X_train_p = X_train_n @ components
X_test_p = X_test_n @ components

print(f"Reduced to {k} dimensions.")
print()

# ============================================================
# BALANCED TRAINING
# ============================================================

def make_balanced(X_data, y_data, class_values, rng):

    groups = []

    for value in class_values:

        indices = np.where(y_data == value)[0]

        if len(indices) == 0:
            continue

        groups.append((value, indices))

    if not groups:
        raise RuntimeError("No training samples found.")

    target_size = max(len(indices) for _, indices in groups)

    balanced_indices = []

    for value, indices in groups:

        chosen = rng.choice(
            indices,
            size=target_size,
            replace=True
        )

        balanced_indices.extend(chosen)

    balanced_indices = np.asarray(
        balanced_indices,
        dtype=np.int64
    )

    rng.shuffle(balanced_indices)

    return (
        X_data[balanced_indices],
        y_data[balanced_indices]
    )


# ============================================================
# SOFTMAX
# ============================================================

def softmax(z):

    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(
        exp_z,
        axis=1,
        keepdims=True
    )


def train_classifier(
    X_data,
    y_data,
    class_values,
    rng
):

    X_bal, y_bal = make_balanced(
        X_data,
        y_data,
        class_values,
        rng
    )

    print("  Balanced training set:")

    for value in class_values:

        count = np.sum(y_bal == value)

        print(
            f"    class {value}: {count}"
        )

    n = X_bal.shape[0]
    d = X_bal.shape[1]
    c = len(class_values)

    W = np.zeros(
        (c, d),
        dtype=np.float32
    )

    b = np.zeros(
        c,
        dtype=np.float32
    )

    class_to_index = {
        value: i
        for i, value in enumerate(class_values)
    }

    targets = np.asarray([
        class_to_index[y]
        for y in y_bal
    ])

    # Small randomized initialization prevents symmetry.
    W += rng.normal(
        0,
        0.005,
        size=W.shape
    ).astype(np.float32)

    for epoch in range(EPOCHS):

        logits = X_bal @ W.T + b

        probs = softmax(logits)

        # Cross entropy gradient.
        grad = probs.copy()

        grad[
            np.arange(n),
            targets
        ] -= 1.0

        grad /= n

        grad_W = grad.T @ X_bal
        grad_b = grad.sum(axis=0)

        # L2 regularization.
        grad_W += L2 * W

        W -= LEARNING_RATE * grad_W
        b -= LEARNING_RATE * grad_b

    return W.astype(np.float32), b.astype(np.float32)


def predict(X_data, W, b):

    logits = X_data @ W.T + b

    probs = softmax(logits)

    predictions = np.argmax(
        probs,
        axis=1
    )

    confidence = np.max(
        probs,
        axis=1
    )

    return predictions, confidence


# ============================================================
# BUILD TARGETS
# ============================================================

def map_labels(labels_in, mapping):

    return np.asarray([
        mapping.get(str(label), 0)
        for label in labels_in
    ], dtype=np.int64)


y_forward_train = map_labels(
    y_train_labels,
    FORWARD_LABELS
)

y_forward_test = map_labels(
    y_test_labels,
    FORWARD_LABELS
)

y_turn_train = map_labels(
    y_train_labels,
    TURN_LABELS
)

y_turn_test = map_labels(
    y_test_labels,
    TURN_LABELS
)

y_camera_train = map_labels(
    y_train_labels,
    CAMERA_LABELS
)

y_camera_test = map_labels(
    y_test_labels,
    CAMERA_LABELS
)

# ============================================================
# TRAIN
# ============================================================

print("=" * 70)
print("TRAINING BALANCED DECODERS")
print("=" * 70)
print()

print("Forward decoder")
print("-" * 40)

W_fwd, b_fwd = train_classifier(
    X_train_p,
    y_forward_train,
    [0, 1, 2],
    rng
)

print()

print("Turn decoder")
print("-" * 40)

W_trn, b_trn = train_classifier(
    X_train_p,
    y_turn_train,
    [0, 1, 2],
    rng
)

print()

print("Camera decoder")
print("-" * 40)

W_cam, b_cam = train_classifier(
    X_train_p,
    y_camera_train,
    [0, 1, 2],
    rng
)

# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    name,
    X_data,
    y_data,
    W,
    b,
    label_names
):

    pred, conf = predict(
        X_data,
        W,
        b
    )

    accuracy = np.mean(
        pred == y_data
    )

    print()
    print(
        f"{name:8s}: "
        f"test={accuracy * 100:.1f}%"
    )

    for class_id, class_name in enumerate(label_names):

        mask = y_data == class_id

        if np.any(mask):

            class_acc = np.mean(
                pred[mask] == class_id
            )

            print(
                f"  {class_name:10s}: "
                f"{class_acc * 100:.1f}% "
                f"({np.sum(mask)} samples)"
            )

    return accuracy


print()
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

forward_acc = evaluate(
    "forward",
    X_test_p,
    y_forward_test,
    W_fwd,
    b_fwd,
    ["none", "forward", "back"]
)

turn_acc = evaluate(
    "turn",
    X_test_p,
    y_turn_test,
    W_trn,
    b_trn,
    ["none", "left", "right"]
)

camera_acc = evaluate(
    "camera",
    X_test_p,
    y_camera_test,
    W_cam,
    b_cam,
    ["none", "cam_left", "cam_right"]
)

# ============================================================
# SAVE
# ============================================================

np.savez_compressed(
    OUTPUT_FILE,

    mean=(
        mean + pca_mean * std
    ).astype(np.float32),

    std=std.astype(np.float32),

    components=components.astype(np.float32),

    weights_forward=W_fwd,
    bias_forward=b_fwd,

    weights_turn=W_trn,
    bias_turn=b_trn,

    weights_camera=W_cam,
    bias_camera=b_cam,

    labels=np.asarray(
        ["none", "forward", "back"]
    ),

    training_seed=np.asarray(
        [RANDOM_SEED]
    )
)

print()
print("=" * 70)
print("DECODER SAVED")
print("=" * 70)
print()
print(f"File: {OUTPUT_FILE}")
print()
print("Training complete.")