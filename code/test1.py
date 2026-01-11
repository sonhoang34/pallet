import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
K = np.array([
    [609.4961547851562, 0.0, 324.0486145019531],
    [0.0, 609.9985961914062, 231.71307373046875],
    [0.0, 0.0, 1.0]
])

y_ground = 0.3

# ===== HELPER FUNCTIONS =====


def pixel_to_ray(u, v, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    d = np.array([x, y, 1.0])
    return d / np.linalg.norm(d)


def intersect_ground(ray_dir, y_ground):
    dy = ray_dir[1]
    if abs(dy) < 1e-6:
        return None
    t = y_ground / dy
    if t <= 0:
        return None
    x = t * ray_dir[0]
    z = t * ray_dir[2]
    return np.array([x, z])

# ===== WEIGHT METHODS =====


def compute_weights_old(points_xz):
    """OLD: Weight = 1/Z² (absolute distance only)"""
    Z_values = points_xz[:, 1]
    weights = 1.0 / (Z_values ** 2)
    weights = weights / weights.sum()
    return weights


def compute_weights_adjacent_delta(points_xz):
    """
    NEW METHOD 1: Weight dựa trên ΔZ với điểm liền kề
    Điểm có ΔZ lớn (jump) → weight thấp
    """
    Z_values = points_xz[:, 1]
    N = len(Z_values)

    # Tính ΔZ liền kề
    delta_Z = np.abs(np.diff(Z_values))

    # Median ΔZ (expected smooth change)
    median_delta = np.median(delta_Z)

    weights = np.ones(N)

    # Điểm đầu
    deviation_0 = abs(delta_Z[0] - median_delta)
    weights[0] = np.exp(-deviation_0 / (median_delta + 1e-6) * 5)

    # Điểm giữa
    for i in range(1, N-1):
        delta_left = delta_Z[i-1]
        delta_right = delta_Z[i]

        # Deviation từ median
        dev_left = abs(delta_left - median_delta)
        dev_right = abs(delta_right - median_delta)

        # Penalty if jumpy
        max_dev = max(dev_left, dev_right)
        weights[i] = np.exp(-max_dev / (median_delta + 1e-6) * 5)

    # Điểm cuối
    deviation_n = abs(delta_Z[-1] - median_delta)
    weights[-1] = np.exp(-deviation_n / (median_delta + 1e-6) * 5)

    # Normalize
    weights = weights / weights.sum()

    return weights


def compute_weights_gradient(points_xz):
    """
    NEW METHOD 2: Weight dựa trên gradient dZ/dX consistency
    """
    X_values = points_xz[:, 0]
    Z_values = points_xz[:, 1]
    N = len(Z_values)

    # Global gradient
    dX_total = X_values[-1] - X_values[0]
    dZ_total = Z_values[-1] - Z_values[0]

    if abs(dX_total) < 1e-6:
        # Vertical line - use ΔZ method instead
        return compute_weights_adjacent_delta(points_xz)

    global_gradient = dZ_total / dX_total

    # Local gradients
    weights = np.ones(N)

    for i in range(N-1):
        dX = X_values[i+1] - X_values[i]
        dZ = Z_values[i+1] - Z_values[i]

        if abs(dX) > 1e-6:
            local_gradient = dZ / dX
            deviation = abs(local_gradient - global_gradient)

            # Exponential penalty
            factor = np.exp(-deviation * 10)
            weights[i] *= factor
            weights[i+1] *= factor

    weights = weights / weights.sum()

    return weights


def compute_weights_line_distance(points_xz):
    """
    NEW METHOD 3: Weight dựa trên khoảng cách đến best-fit line
    """
    # Quick PCA
    mean = points_xz.mean(axis=0)
    centered = points_xz - mean

    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    direction = eigenvectors[:, np.argmax(eigenvalues)].real

    # Normal vector
    normal = np.array([-direction[1], direction[0]])

    # Distance to line
    distances = np.abs((points_xz - mean) @ normal)

    # Robust scale estimate
    mad = np.median(distances)
    sigma = max(mad * 1.4826, 0.01)  # MAD to std conversion

    # Gaussian kernel
    weights = np.exp(-(distances / sigma) ** 2)
    weights = weights / weights.sum()

    return weights


def compute_weights_combined(points_xz):
    """
    BEST: Combine multiple factors
    """
    # Factor 1: ΔZ consistency (most important)
    w_delta = compute_weights_adjacent_delta(points_xz)

    # Factor 2: Line distance
    w_dist = compute_weights_line_distance(points_xz)

    # Factor 3: Mild Z penalty (much softer than old method)
    Z_values = points_xz[:, 1]
    w_z = 1.0 / (Z_values ** 1.0)  # Linear, not quadratic
    w_z = w_z / w_z.sum()

    # Combine: multiply = log-add
    # Priority: delta > distance > z_penalty
    weights = (w_delta ** 2) * (w_dist ** 1.5) * (w_z ** 0.5)
    weights = weights / weights.sum()

    return weights

# ===== WEIGHTED PCA =====


def weighted_pca_yaw(points_xz, weights):
    """Compute yaw using weighted PCA"""
    # Weighted mean
    mean_w = np.average(points_xz, axis=0, weights=weights)

    # Centered
    centered = points_xz - mean_w

    # Weighted covariance
    cov_w = (centered.T * weights) @ centered

    # Eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(cov_w)
    direction = eigenvectors[:, np.argmax(eigenvalues)].real

    if direction[0] < 0:
        direction = -direction

    yaw = np.arctan2(direction[0], direction[1]) - np.pi/2

    return yaw

# ===== COMPARISON FRAMEWORK =====


def compare_weight_methods(xs, ys, K, y_ground, true_yaw=None):
    """
    So sánh tất cả các phương pháp
    """
    # Back-project
    points_xz = []
    for x, y in zip(xs, ys):
        ray = pixel_to_ray(x, y, K)
        p = intersect_ground(ray, y_ground)
        if p is not None:
            points_xz.append(p)

    if len(points_xz) < 5:
        print("Not enough points")
        return None

    points_xz = np.array(points_xz)

    print("=" * 70)
    print("WEIGHT METHOD COMPARISON")
    print("=" * 70)
    print(f"Number of points: {len(points_xz)}")
    print(
        f"Z range: [{points_xz[:, 1].min():.2f}, {points_xz[:, 1].max():.2f}]m")

    if true_yaw is not None:
        print(f"True yaw: {np.rad2deg(true_yaw):.2f}°")
    print()

    # Unweighted baseline
    print("0. UNWEIGHTED (Baseline)")
    mean_unw = points_xz.mean(axis=0)
    centered = points_xz - mean_unw
    cov = centered.T @ centered
    eig_val, eig_vec = np.linalg.eig(cov)
    direction = eig_vec[:, np.argmax(eig_val)].real
    if direction[0] < 0:
        direction = -direction
    yaw_unw = np.arctan2(direction[0], direction[1]) - np.pi/2
    print(f"   Yaw: {np.rad2deg(yaw_unw):7.2f}°")
    if true_yaw is not None:
        print(
            f"   Error: {abs(np.rad2deg(yaw_unw) - np.rad2deg(true_yaw)):.2f}°")

    # Old method
    print("\n1. OLD METHOD (weight = 1/Z²)")
    w_old = compute_weights_old(points_xz)
    yaw_old = weighted_pca_yaw(points_xz, w_old)
    print(f"   Yaw: {np.rad2deg(yaw_old):7.2f}°")
    print(f"   Weight ratio: {w_old.max()/w_old.min():.2f}x")
    if true_yaw is not None:
        print(
            f"   Error: {abs(np.rad2deg(yaw_old) - np.rad2deg(true_yaw)):.2f}°")

    # New methods
    methods = {
        'Adjacent ΔZ': compute_weights_adjacent_delta,
        'Gradient': compute_weights_gradient,
        'Line Distance': compute_weights_line_distance,
        'Combined': compute_weights_combined
    }

    results = {}

    for i, (name, func) in enumerate(methods.items(), start=2):
        print(f"\n{i}. NEW METHOD: {name}")
        weights = func(points_xz)
        yaw = weighted_pca_yaw(points_xz, weights)

        results[name] = {'yaw': yaw, 'weights': weights}

        print(f"   Yaw: {np.rad2deg(yaw):7.2f}°")
        print(f"   Weight ratio: {weights.max()/weights.min():.2f}x")

        if true_yaw is not None:
            error = abs(np.rad2deg(yaw) - np.rad2deg(true_yaw))
            print(f"   Error: {error:.2f}°")

        # Weight statistics
        Z_vals = points_xz[:, 1]
        near_mask = Z_vals < np.median(Z_vals)
        weight_near = weights[near_mask].sum()
        print(f"   Near points weight: {weight_near*100:.1f}%")

    # Best method
    if true_yaw is not None:
        print("\n" + "=" * 70)
        print("BEST METHOD")
        print("=" * 70)

        errors = {
            'Unweighted': abs(np.rad2deg(yaw_unw) - np.rad2deg(true_yaw)),
            'Old (1/Z²)': abs(np.rad2deg(yaw_old) - np.rad2deg(true_yaw))
        }

        for name, data in results.items():
            errors[name] = abs(np.rad2deg(data['yaw']) - np.rad2deg(true_yaw))

        best_method = min(errors.items(), key=lambda x: x[1])
        print(f"✅ {best_method[0]}: {best_method[1]:.3f}° error")

    return points_xz, yaw_unw, yaw_old, results

# ===== VISUALIZATION =====


def visualize_consistency_weights(points_xz, true_yaw=None):
    """
    Visualize tất cả weight methods
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    Z_values = points_xz[:, 1]

    # Method configs
    methods = [
        ('Unweighted', None, (0, 0)),
        ('Old: 1/Z²', compute_weights_old, (0, 1)),
        ('New: Adjacent ΔZ', compute_weights_adjacent_delta, (0, 2)),
        ('New: Gradient', compute_weights_gradient, (1, 0)),
        ('New: Line Distance', compute_weights_line_distance, (1, 1)),
        ('New: Combined', compute_weights_combined, (1, 2))
    ]

    yaws = {}

    for name, func, (row, col) in methods:
        ax = fig.add_subplot(gs[row, col])

        if func is None:
            # Unweighted
            weights = np.ones(len(points_xz)) / len(points_xz)
        else:
            weights = func(points_xz)

        # Compute yaw
        yaw = weighted_pca_yaw(points_xz, weights)
        yaws[name] = yaw

        # Plot points with size = weight
        sizes = weights * 2000

        scatter = ax.scatter(points_xz[:, 0], points_xz[:, 1],
                             c=weights, s=sizes, alpha=0.6,
                             cmap='RdYlGn', edgecolors='black', linewidths=1,
                             vmin=0, vmax=weights.max())

        # Fit line
        mean_w = np.average(points_xz, axis=0, weights=weights)
        centered = points_xz - mean_w
        cov_w = (centered.T * weights) @ centered
        eig_val, eig_vec = np.linalg.eig(cov_w)
        direction = eig_vec[:, np.argmax(eig_val)].real
        if direction[0] < 0:
            direction = -direction

        t = np.linspace(-0.8, 0.8, 100)
        line = mean_w + np.outer(t, direction)
        ax.plot(line[:, 0], line[:, 1], 'r-', linewidth=2)

        # Title with error
        title = f'{name}\nYaw: {np.rad2deg(yaw):.1f}°'
        if true_yaw is not None:
            error = abs(np.rad2deg(yaw) - np.rad2deg(true_yaw))
            title += f'\nError: {error:.2f}°'

        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Z (m)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Colorbar
        plt.colorbar(scatter, ax=ax, label='Weight')

    # Bottom row: Weight analysis

    # ΔZ plot
    ax_delta = fig.add_subplot(gs[2, 0])
    delta_Z = np.abs(np.diff(Z_values))
    indices = np.arange(len(delta_Z))

    ax_delta.bar(indices, delta_Z * 100, alpha=0.6, edgecolor='black')
    ax_delta.axhline(y=np.median(delta_Z)*100, color='r', linestyle='--',
                     linewidth=2, label=f'Median: {np.median(delta_Z)*100:.1f}cm')
    ax_delta.set_xlabel('Point index', fontsize=10)
    ax_delta.set_ylabel('|ΔZ| between adjacent (cm)', fontsize=10)
    ax_delta.set_title('Adjacent ΔZ Analysis\n(Jump detection)',
                       fontsize=11, fontweight='bold')
    ax_delta.grid(True, alpha=0.3, axis='y')
    ax_delta.legend()

    # Weight comparison
    ax_weights = fig.add_subplot(gs[2, 1])

    w_old = compute_weights_old(points_xz)
    w_new = compute_weights_combined(points_xz)

    x_pos = np.arange(len(points_xz))
    width = 0.35

    ax_weights.bar(x_pos - width/2, w_old, width,
                   label='Old (1/Z²)', alpha=0.7)
    ax_weights.bar(x_pos + width/2, w_new, width,
                   label='New (Combined)', alpha=0.7)

    ax_weights.set_xlabel('Point index', fontsize=10)
    ax_weights.set_ylabel('Weight', fontsize=10)
    ax_weights.set_title('Weight Comparison', fontsize=11, fontweight='bold')
    ax_weights.legend()
    ax_weights.grid(True, alpha=0.3, axis='y')

    # Yaw comparison
    ax_yaw = fig.add_subplot(gs[2, 2])

    method_names = list(yaws.keys())
    yaw_values = [np.rad2deg(yaws[m]) for m in method_names]

    colors = ['gray', 'red', 'green', 'blue', 'orange', 'purple']
    bars = ax_yaw.barh(method_names, yaw_values, color=colors[:len(
        method_names)], alpha=0.7, edgecolor='black')

    if true_yaw is not None:
        ax_yaw.axvline(x=np.rad2deg(true_yaw), color='black', linestyle='--',
                       linewidth=2, label=f'True: {np.rad2deg(true_yaw):.1f}°')

    ax_yaw.set_xlabel('Yaw (degrees)', fontsize=10)
    ax_yaw.set_title('Yaw Comparison', fontsize=11, fontweight='bold')
    ax_yaw.grid(True, alpha=0.3, axis='x')
    if true_yaw is not None:
        ax_yaw.legend()

    # Add value labels
    for bar, val in zip(bars, yaw_values):
        width = bar.get_width()
        ax_yaw.text(width, bar.get_y() + bar.get_height()/2, f'{val:.1f}°',
                    ha='left', va='center', fontsize=9, fontweight='bold')

    plt.savefig('consistency_weights_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to 'consistency_weights_comparison.png'")
    plt.show()

# ===== DEMO =====


if __name__ == "__main__":
    print("CONSISTENCY-BASED WEIGHT DEMONSTRATION\n")

    # Simulate realistic scenario: line with some outliers
    np.random.seed(42)

    true_angle = 30  # degrees
    n_points = 20

    # Generate clean line
    t_values = np.linspace(0, 1, n_points)
    X_clean = t_values * 1.2 * np.cos(np.deg2rad(true_angle))
    Z_clean = 1.0 + t_values * 1.0  # 1m to 2m

    # Add realistic errors
    X_noisy = X_clean.copy()
    Z_noisy = Z_clean.copy()

    for i in range(n_points):
        # Base error ∝ Z²
        error_scale = (Z_clean[i] / 1.0) ** 2 * 0.03

        X_noisy[i] += np.random.normal(0, error_scale * 0.5)
        Z_noisy[i] += np.random.normal(0, error_scale)

    # Add outliers (jumps)
    outlier_indices = [5, 12, 17]
    for idx in outlier_indices:
        Z_noisy[idx] += np.random.choice([-1, 1]) * 0.15  # 15cm jump!

    print(f"True yaw: {true_angle}°")
    print(f"Points: {n_points} (with {len(outlier_indices)} outliers)")
    print(f"Outliers at indices: {outlier_indices}")
    print(f"Z range: {Z_clean.min():.2f}m to {Z_clean.max():.2f}m\n")

    # Project to 2D
    xs_sim = []
    ys_sim = []

    for X, Z in zip(X_noisy, Z_noisy):
        Y_3d = -y_ground
        u = K[0, 0] * (X / Z) + K[0, 2]
        v = K[1, 1] * (Y_3d / Z) + K[1, 2]
        xs_sim.append(int(u))
        ys_sim.append(int(v))

    xs_sim = np.array(xs_sim)
    ys_sim = np.array(ys_sim)

    # Build 3D points for direct testing
    points_xz = np.column_stack([X_noisy, Z_noisy])

    print(points_xz)
    
    # Compare methods
    points, yaw_unw, yaw_old, results = compare_weight_methods(
        xs_sim, ys_sim, K, y_ground, np.deg2rad(true_angle)
    )
    
    # Visualize
    visualize_consistency_weights(points_xz, np.deg2rad(true_angle))

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("""
✅ BEST METHOD: Combined (Adjacent ΔZ + Line Distance + mild Z penalty)

Tại sao?
1. Phát hiện outliers/jumps tốt nhất (qua Adjacent ΔZ)
2. Robust với noise (qua Line Distance)
3. Vẫn giữ mild penalty cho điểm xa
4. Không bias quá mạnh về Z như old method

Code để dùng:
    weights = compute_weights_combined(points_xz)
    yaw = weighted_pca_yaw(points_xz, weights)
    """)
