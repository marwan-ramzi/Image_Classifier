def reward_function(params):
    # Example of penalize steering, which helps mitigate zig-zag behaviors

    # Read input parameters
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    abs_steering = abs(params['steering_angle']) # Only need the absolute steering angle
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    
    # Calculate 3 marks that are farther and father away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Set the speed threshold based your action space
    SPEED_THRESHOLD = 1.5

    if not all_wheels_on_track:
        # Penalize if the car goes off track
        reward = 1e-3
    elif (speed < SPEED_THRESHOLD) & (distance_from_center <= marker_1):
        # Penalize if the car goes too slow
        reward = 0.1
    elif (speed < SPEED_THRESHOLD) & (distance_from_center <= marker_2):
        # Penalize if the car goes too slow
        reward = 0.25
    elif (speed < SPEED_THRESHOLD) & (distance_from_center <= marker_3):
        # Penalize if the car goes too slow
        reward = 0.5
    else:
        # High reward if the car stays on track and goes fast
        reward = 1.0

     # Steering penality threshold, change the number based on your action space setting
    ABS_STEERING_THRESHOLD = 20 
    
    # Penalize reward if the car is steering too much
    if abs_steering > ABS_STEERING_THRESHOLD:
        reward *= 0.8
        
    return float(reward)

    # # Give higher reward if the car is closer to center line and vice versa
    # if distance_from_center <= marker_1:
    #     reward = 1.0
    # elif distance_from_center <= marker_2:
    #     reward = 0.5
    # elif distance_from_center <= marker_3:
    #     reward = 0.1
    # else:
    #     reward = 1e-3  # likely crashed/ close to off track
