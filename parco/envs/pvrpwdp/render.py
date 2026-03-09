import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib import cm, colormaps

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None):
    """Render PVRPWDPV2 environment với multi-vehicle (trucks + drones).
    Chia màu theo từng xe (vehicle), không theo route.
    """
    # Get vehicle info
    num_trucks = td["num_trucks"]
    num_drones = td["num_drones"]
    if num_trucks.dim() > 0:
        num_trucks = num_trucks[0].item() if td.batch_size != torch.Size([]) else num_trucks.item()
    else:
        num_trucks = num_trucks.item()
    if num_drones.dim() > 0:
        num_drones = num_drones[0].item() if td.batch_size != torch.Size([]) else num_drones.item()
    else:
        num_drones = num_drones.item()
    
    num_vehicles = num_trucks + num_drones
    
    # Tạo colormap cho vehicles (không phải routes)
    base = colormaps["nipy_spectral"]
    color_list = base(np.linspace(0, 1, num_vehicles))
    cmap_name = base.name + str(num_vehicles)
    out = base.from_list(cmap_name, color_list, num_vehicles)

    if ax is None:
        _, ax = plt.subplots()

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)

    # Handle batch dimension
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    locs = td["locs"]
    scale_demand = td["capacity_per_vehicle"].max()
    demands = td["demand"] * scale_demand

    if actions is None:
        log.warning("No action in TensorDict, rendering unsorted locs")
    else:
        # Thêm depot vào đầu và cuối
        actions = torch.cat([torch.tensor([0]), actions, torch.tensor([0])])

    x, y = locs[:, 0], locs[:, 1]

    # Plot depot
    ax.scatter(
        locs[0, 0],
        locs[0, 1],
        edgecolors=cm.Set2(2),
        facecolors="none",
        s=100,
        linewidths=2,
        marker="s",
        alpha=1,
    )

    # Plot customers
    ax.scatter(
        x[1:],
        y[1:],
        edgecolors=cm.Set2(0),
        facecolors="none",
        s=50,
        linewidths=2,
        marker="o",
        alpha=1,
    )

    # Plot demand bars
    for node_idx in range(1, len(locs)):
        ax.add_patch(
            plt.Rectangle(
                (locs[node_idx, 0] - 0.005, locs[node_idx, 1] + 0.015),
                0.01,
                demands[node_idx] / (scale_demand * 10),
                edgecolor=cm.Set2(0),
                facecolor=cm.Set2(0),
                fill=True,
            )
        )

    # Text demand
    for node_idx in range(1, len(locs)):
        ax.text(
            locs[node_idx, 0],
            locs[node_idx, 1] - 0.025,
            f"{demands[node_idx].item():.2f}",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=10,
            color=cm.Set2(0),
        )

    # Text depot
    ax.text(
        locs[0, 0],
        locs[0, 1] - 0.025,
        "Depot",
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10,
        color=cm.Set2(2),
    )

    # Plot actions - chia màu theo vehicle
    if actions is not None:
        vehicle_idx = 0
        for action_idx in range(len(actions) - 1):
            # Mỗi lần về depot = chuyển sang xe mới
            if actions[action_idx] == 0 and action_idx > 0:
                vehicle_idx += 1
                if vehicle_idx >= num_vehicles:
                    vehicle_idx = 0  # Wrap around nếu vượt quá số xe
            
            from_loc = locs[actions[action_idx]]
            to_loc = locs[actions[action_idx + 1]]
            
            # Xác định style dựa trên loại xe
            is_truck = vehicle_idx < num_trucks
            linestyle = "-" if is_truck else "--"
            linewidth = 1.5 if is_truck else 1
            
            ax.plot(
                [from_loc[0], to_loc[0]],
                [from_loc[1], to_loc[1]],
                color=out(vehicle_idx),
                lw=linewidth,
                linestyle=linestyle,
            )
            ax.annotate(
                "",
                xy=(to_loc[0], to_loc[1]),
                xytext=(from_loc[0], from_loc[1]),
                arrowprops=dict(
                    arrowstyle="-|>", 
                    color=out(vehicle_idx),
                    linestyle=linestyle,
                ),
                size=15,
                annotation_clip=False,
            )

    return ax
