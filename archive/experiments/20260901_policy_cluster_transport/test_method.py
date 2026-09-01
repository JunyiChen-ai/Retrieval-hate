import torch

from method import PolicyClusterTransport


def run():
    torch.manual_seed(3)
    shared = torch.randn(3, 12, 8, requires_grad=True)
    primitive = torch.randn(3, 12, 6, requires_grad=True)
    raw = torch.randn(3, 12, requires_grad=True)
    frame = torch.sigmoid(raw)
    valid = torch.tensor([[1] * 12, [1] * 9 + [0] * 3,
                          [1] * 12], dtype=torch.bool)
    labels = torch.tensor([0., 1., 1.])
    # A high lower bound forces the projection branch, including its tensor
    # broadcasting path, without running any training.
    module = PolicyClusterTransport(8, .2, min_harmful_mass=.95)
    loss, stats = module(shared, primitive, frame, valid, labels,
                         "hatemm", "policy")
    assert torch.isfinite(loss) and stats["harmful_mass"] > 0
    loss.backward()
    assert shared.grad is not None and primitive.grad is not None
    for arm in ("binary", "permuted"):
        value, _ = module(shared.detach(), primitive.detach(), frame.detach(),
                          valid, labels, "hateclipseg", arm)
        assert torch.isfinite(value)

    # Result-affecting regression: constrained targets must train the shared
    # representation even when every bag is negative (background anchoring)
    # and for HCS positives, whose policy-valid set spans all primitive states.
    for check_labels, corpus in ((torch.zeros(2), "hatemm"),
                                 (torch.ones(2), "hateclipseg")):
        check_shared = torch.randn(2, 7, 8, requires_grad=True)
        check_primitive = torch.randn(2, 7, 6, requires_grad=True)
        check_frame = torch.sigmoid(torch.randn(2, 7, requires_grad=True))
        check_valid = torch.ones(2, 7, dtype=torch.bool)
        check_module = PolicyClusterTransport(8, .2)
        check_loss, _ = check_module(
            check_shared, check_primitive, check_frame, check_valid,
            check_labels, corpus, "policy")
        check_loss.backward()
        assert check_shared.grad is not None
        assert float(check_shared.grad.abs().sum()) > 0
    print("PASS")


if __name__ == "__main__":
    run()
