
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import argparse

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def vectors_two_axes(theta_x_deg, theta_y_deg, order="XY"):
    """
    Returns (v1, v2) unit vectors for the two joint 'aim' directions.
    v1 = forward ez after first rotation only.
    v2 = forward ez after both rotations.
    order="XY" => R = Ry(θy) * Rx(θx), so first rotation is Rx.
    order="YX" => R = Rx(θx) * Ry(θy), so first rotation is Ry.
    """
    ez = np.array([0.0, 0.0, 1.0])
    tx = np.deg2rad(theta_x_deg)
    ty = np.deg2rad(theta_y_deg)
    if order.upper() == "XY":
        R1 = rot_x(tx)
        R2 = rot_y(ty)
        v1 = R1 @ ez
        v2 = (R2 @ R1) @ ez
    elif order.upper() == "YX":
        R1 = rot_y(ty)
        R2 = rot_x(tx)
        v1 = R1 @ ez
        v2 = (R2 @ R1) @ ez
    else:
        raise ValueError("order must be 'XY' or 'YX'")
    return v1, v2

def plot_two_vectors(theta_x_deg, theta_y_deg, order="XY", len1=1.0, len2=1.0, chain=False, show=True, ax=None):
    v1, v2 = vectors_two_axes(theta_x_deg, theta_y_deg, order=order)
    p0 = np.zeros(3)
    p1 = p0 + len1 * v1
    if chain:
        # draw second from tip of first (serial 2-link look)
        p2_start = p1
    else:
        p2_start = p0
    p2 = p2_start + len2 * v2

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # World axes (defaults)
    ax.quiver(0,0,0, 1,0,0, length=1.0, arrow_length_ratio=0.08)
    ax.quiver(0,0,0, 0,1,0, length=1.0, arrow_length_ratio=0.08)
    ax.quiver(0,0,0, 0,0,1, length=1.0, arrow_length_ratio=0.08)

    # First joint vector
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], label="Joint 1 vector")
    ax.scatter([p1[0]], [p1[1]], [p1[2]])

    # Second joint vector
    ax.plot([p2_start[0], p2[0]], [p2_start[1], p2[1]], [p2_start[2], p2[2]], label="Joint 2 vector")
    ax.scatter([p2[0]], [p2[1]], [p2[2]])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title_chain = "chained" if chain else "overlayed at origin"
    ax.set_title(f"Two Joint Vectors ({title_chain}) — order {order}, θx={theta_x_deg:.1f}°, θy={theta_y_deg:.1f}°")
    lim = max(1.2, len1 + len2 + 0.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    try:
        ax.set_box_aspect([1,1,1])
    except Exception:
        pass
    ax.legend(loc="upper left")

    if show:
        plt.show()
    return ax

def launch_sliders(order="XY", len1=1.0, len2=1.0, chain=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.1, bottom=0.30)

    theta_x0, theta_y0 = 0.0, 0.0
    plot_two_vectors(theta_x0, theta_y0, order=order, len1=len1, len2=len2, chain=chain, show=False, ax=ax)

    ax_x = plt.axes([0.1, 0.20, 0.8, 0.03])
    ax_y = plt.axes([0.1, 0.15, 0.8, 0.03])
    ax_l1 = plt.axes([0.1, 0.10, 0.35, 0.03])
    ax_l2 = plt.axes([0.55, 0.10, 0.35, 0.03])
    ax_chk = plt.axes([0.1, 0.04, 0.25, 0.05])

    s_x = Slider(ax_x, "θx (deg)", -90.0, 90.0, valinit=theta_x0)
    s_y = Slider(ax_y, "θy (deg)", -90.0, 90.0, valinit=theta_y0)
    s_L1 = Slider(ax_l1, "len1", 0.2, 2.5, valinit=len1)
    s_L2 = Slider(ax_l2, "len2", 0.2, 2.5, valinit=len2)

    chk = CheckButtons(ax_chk, ["chain"], [chain])

    def update(val):
        ax.cla()
        plot_two_vectors(s_x.val, s_y.val, order=order, len1=s_L1.val, len2=s_L2.val,
                         chain=chk.get_status()[0], show=False, ax=ax)
        fig.canvas.draw_idle()

    s_x.on_changed(update)
    s_y.on_changed(update)
    s_L1.on_changed(update)
    s_L2.on_changed(update)
    chk.on_clicked(lambda _: update(None))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Two-vector visualizer for a 2-DoF Fable-like joint")
    parser.add_argument("--theta-x", type=float, default=0.0, help="Rotation about X in degrees")
    parser.add_argument("--theta-y", type=float, default=0.0, help="Rotation about Y in degrees")
    parser.add_argument("--order", type=str, default="XY", choices=["XY","YX"],
                        help="Rotation order: XY means R = Ry(θy) * Rx(θx)")
    parser.add_argument("--len1", type=float, default=1.0, help="Length of joint 1 vector")
    parser.add_argument("--len2", type=float, default=1.0, help="Length of joint 2 vector")
    parser.add_argument("--chain", action="store_true", help="Draw joint 2 from the tip of joint 1")
    parser.add_argument("--sliders", action="store_true", help="Launch interactive sliders")

    args = parser.parse_args()

    if args.sliders:
        launch_sliders(order=args.order, len1=args.len1, len2=args.len2, chain=args.chain)
    else:
        plot_two_vectors(args.theta_x, args.theta_y, order=args.order, len1=args.len1, len2=args.len2, chain=args.chain)

if __name__ == "__main__":
    main()
