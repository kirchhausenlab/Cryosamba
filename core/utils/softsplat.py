import torch

grid_cache = {}


def get_cached_grid(H, W, device):
    key = (H, W, device)
    if key not in grid_cache:
        gridY, gridX = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        grid = (
            torch.stack((gridX, gridY), dim=0)
            .unsqueeze(1)
            .unsqueeze(1)
            .view(2, 1, 1, H * W)
        )
        plus_one = (
            torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]], device=device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # 2,4,1,1,1
        grid_cache[key] = (grid, plus_one)
    return grid_cache[key]


def func_softsplat(tenInput, tenFlow):
    (B, C, H, W), device = tenInput.shape, tenInput.device

    tenInput = tenInput.unsqueeze(0).view(1, B, C, H * W)
    tenFlow = tenFlow.permute(1, 0, 2, 3).view(2, B, 1, H * W)
    tenOutput = torch.zeros((4, B, C, H * W), device=device, dtype=tenInput.dtype)

    grid, plus_one = get_cached_grid(H, W, device)
    flt = grid + tenFlow  # [2,B,1,H*W]

    flt_floor = torch.floor(flt).long()

    coords = flt_floor.unsqueeze(1) + plus_one
    indices = (
        coords[1].clamp(0, H - 1) * W + coords[0].clamp(0, W - 1)
    ).repeat_interleave(C, dim=2)

    tX, tY = (flt - flt_floor).split(1, dim=0)
    tXY = tX * tY
    weights = torch.cat([(1 - tX - tY + tXY), (tX - tXY), (tY - tXY), tXY], dim=0)

    tenOutput.scatter_add_(-1, indices, tenInput * weights)

    return tenOutput.sum(dim=0).reshape(B, C, H, W)


def FunctionSoftsplat(tenInput, tenFlow, tenMetric=None, strType="average"):
    assert tenMetric is None or tenMetric.shape[1] == 1
    assert strType in ["summation", "average", "linear", "softmax"]

    if strType == "average":
        tenInput = torch.cat(
            [
                tenInput,
                tenInput.new_ones(
                    tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3]
                ),
            ],
            1,
        )

    elif strType == "linear":
        tenInput = torch.cat([tenInput * tenMetric, tenMetric], 1)

    elif strType == "softmax":
        tenInput = torch.cat([tenInput * tenMetric.exp(), tenMetric.exp()], 1)

    tenOutput = func_softsplat(tenInput, tenFlow)

    if strType != "summation":
        tenNormalize = tenOutput[:, -1:, :, :]

        tenNormalize[tenNormalize == 0.0] = 1.0

        tenOutput = tenOutput[:, :-1, :, :] / tenNormalize

    return tenOutput
