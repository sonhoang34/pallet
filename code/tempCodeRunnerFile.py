p_start_origin = (longest[0] + x_min, longest[1] + y_min)
    p_end_origin = (longest[2] + x_min, longest[3] + y_min)
    
    num_points = 20
    x_vals = np.linspace(p_start_origin[0], p_end_origin[0], num_points)
    y_vals = np.linspace(p_start_origin[1], p_end_origin[1], num_points)