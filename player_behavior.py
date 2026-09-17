import time





class PlayerBehavior:

    """

    Social behavior layer.



    IMPORTANT:

    This does NOT permanently follow players.



    It:

        1. Detects a player.

        2. Approaches them.

        3. Stops when close enough.

        4. Says hi once.

        5. Inspects them briefly.

        6. Leaves and returns control to the neural brain.

    """



    def __init__(self):

        self.target = None

        self.target_name = None



        self.state = "wandering"



        self.state_started = time.time()



        self.greeted = {}

        self.last_greeting_time = 0.0



        self.approach_timeout = 8.0

        self.inspect_time = 3.0

        self.greeting_cooldown = 12.0



    # ============================================================

    # TARGET

    # ============================================================



    def clear_target(self):

        self.target = None

        self.target_name = None

        self.state = "wandering"

        self.state_started = time.time()



    def set_target(self, player):

        if not player:

            return False



        name = str(

            player.get("name")

            or player.get("player")

            or player.get("username")

            or "Player"

        )



        self.target = player

        self.target_name = name

        self.state = "approaching"

        self.state_started = time.time()



        return True



    # ============================================================

    # GREETING MEMORY

    # ============================================================



    def already_greeted(self, name):

        if not name:

            return False



        t = self.greeted.get(name)



        if t is None:

            return False



        return (time.time() - t) < self.greeting_cooldown



    def mark_greeted(self, name):

        if name:

            self.greeted[name] = time.time()



    # ============================================================

    # STATE

    # ============================================================



    def get_state(self):

        return self.state



    def has_target(self):

        return self.target is not None



    def get_target(self):

        return self.target



    # ============================================================

    # PLAYER POSITION

    # ============================================================



    def get_horizontal_position(self):

        """

        Expected player detection format:



            x = -1.0   far left

            x =  0.0   center

            x = +1.0   far right



        Also accepts:

            center_x

            screen_x

            position

        """



        if not self.target:

            return 0.0



        for key in ("x", "center_x", "screen_x"):

            value = self.target.get(key)



            if isinstance(value, (int, float)):

                x = float(value)



                if x > 1.0:

                    # Raw pixel coordinate (e.g. a 640px-wide frame).

                    x = (x / 640.0) * 2.0 - 1.0

                else:

                    # semantic_vision.detect_players() reports x as a

                    # 0.0-1.0 fraction of screen width, where:

                    #   0.0 = far left, 0.5 = center, 1.0 = far right

                    # Convert that fraction into our -1.0..+1.0,

                    # 0.0-centered convention.

                    x = (x - 0.5) * 2.0



                return max(-1.0, min(1.0, x))



        position = str(

            self.target.get("position", "")

        ).lower()



        if "left" in position:

            return -0.65



        if "right" in position:

            return 0.65



        return 0.0



    # semantic_vision.detect_players() reports "distance" as one

    # of these category strings rather than a number. Map them to

    # representative numeric values on the same 0.0-1.0 scale used

    # by the rest of this class (<=0.20 counts as "close enough").

    DISTANCE_CATEGORIES = {

        "near": 0.15,

        "medium": 0.50,

        "far": 0.90,

    }



    def get_distance(self):

        """

        Accept several possible distance fields.



        Returns:

            approximate distance

            or None when unavailable.

        """



        if not self.target:

            return None



        for key in (

            "distance",

            "distance_estimate",

            "estimated_distance",

        ):

            value = self.target.get(key)



            if isinstance(value, (int, float)):

                return float(value)



            if isinstance(value, str):

                category = self.DISTANCE_CATEGORIES.get(

                    value.strip().lower()

                )



                if category is not None:

                    return category



        return None



    # ============================================================

    # TARGET VALIDITY

    # ============================================================



    def target_is_valid(self):

        if not self.target:

            return False



        confidence = self.target.get("confidence", 1.0)



        try:

            confidence = float(confidence)

        except Exception:

            confidence = 0.0



        return confidence >= 0.35



    # ============================================================

    # BEHAVIOR DECISION

    # ============================================================



    def update(self, detected_players):

        """

        Returns:



            {

                "state": ...,

                "target": ...,

                "x": ...,

                "distance": ...,

                "greet": bool

            }

        """



        now = time.time()



        # --------------------------------------------------------

        # Existing target

        # --------------------------------------------------------



        if self.target is not None:



            if not self.target_is_valid():

                self.clear_target()



            elif now - self.state_started > self.approach_timeout:

                self.clear_target()



            else:

                distance = self.get_distance()



                # If the detector provides distance, use it.

                if distance is not None:



                    if distance <= 0.20:

                        if self.state == "approaching":

                            self.state = "inspecting"

                            self.state_started = now



                    if (

                        self.state == "inspecting"

                        and now - self.state_started >= self.inspect_time

                    ):

                        self.clear_target()



                # Otherwise use a short inspection timeout.

                elif (

                    self.state == "inspecting"

                    and now - self.state_started >= self.inspect_time

                ):

                    self.clear_target()



        # --------------------------------------------------------

        # Acquire a new player

        # --------------------------------------------------------



        if self.target is None and detected_players:



            candidates = []



            for player in detected_players:



                if not isinstance(player, dict):

                    continue



                confidence = player.get("confidence", 0.0)



                try:

                    confidence = float(confidence)

                except Exception:

                    continue



                # Accept weaker but real player detections to prevent the fly
                # from missing a valid target and then oscillating in place.
                if confidence < 0.35:

                    continue



                name = str(

                    player.get("name")

                    or player.get("player")

                    or player.get("username")

                    or "Player"

                )



                if self.already_greeted(name):

                    continue



                candidates.append(

                    (confidence, player)

                )



            if candidates:



                candidates.sort(

                    key=lambda item: item[0],

                    reverse=True

                )



                self.set_target(candidates[0][1])



        # --------------------------------------------------------

        # Greeting

        # --------------------------------------------------------



        greet = False



        if self.target is not None:



            name = self.target_name



            if self.state == "inspecting":



                if not self.already_greeted(name):



                    if now - self.last_greeting_time >= 2.0:



                        greet = True

                        self.mark_greeted(name)

                        self.last_greeting_time = now



        return {

            "state": self.state,

            "target": self.target,

            "x": self.get_horizontal_position(),

            "distance": self.get_distance(),

            "greet": greet,

        }