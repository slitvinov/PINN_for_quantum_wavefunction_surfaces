import torch

def linear(x, A, b):
        return x @ A + b

def d(f, x):
    df, = torch.autograd.grad(torch.sum(f), x, create_graph=True)
    return torch.autograd.grad(torch.sum(df), x, create_graph=True)[0]


def ini(*shape):
    x = torch.empty(shape, dtype=dtype, requires_grad=True)
    L = 1 / shape[0]**0.5
    with torch.no_grad():
        x.uniform_(-L, L)
    return x


def train(params, lr, epochs):
    tt = 0
    optimizer = torch.optim.Adam(params, lr=lr)
    while True:
        optimizer.zero_grad()
        if tt % sc_sampling == 0:
            with torch.no_grad():
                x.uniform_(-L, L)
                y.uniform_(-L, L)
                z.uniform_(-L, L)
                R.uniform_(Rlo, Rhi)
                r1sq = (x - R)**2 + y**2 + z**2
                r2sq = (x + R)**2 + y**2 + z**2
                x[r1sq < cutoff**2] = cutoff
                x[r2sq < cutoff**2] = cutoff
                r1sq = (x - R)**2 + y**2 + z**2
                r2sq = (x + R)**2 + y**2 + z**2
                i1, = torch.where(r1sq[:, 0] >= bcutoff**2)
                i2, = torch.where(r2sq[:, 0] >= bcutoff**2)

        r1 = torch.sqrt((x - R)**2 + y**2 + z**2)
        r2 = torch.sqrt((x + R)**2 + y**2 + z**2)
        f1 = torch.exp(-r1)
        f2 = torch.exp(-r2)
        h = torch.sigmoid(linear(torch.hstack((f1, f2)), H1a, H1b))
        h = torch.sigmoid(linear(h, H2a, H2b))
        h = linear(2 * h, H3a, H3b)
        l = torch.sigmoid(linear(R, L1a, L1b))
        l = linear(l, L2a, L2b)
        e = torch.sigmoid(linear(R, E1a, E1b))
        e = torch.sigmoid(linear(e, E2a, E2b))
        e = linear(e, E3a, E3b)
        psi = f1 + f2 + h * l
        res = d(psi, x) + d(psi, y) + d(psi, z) + (e + 1 / r1 + 1 / r2) * psi
        Lpde = (res**2).mean()
        Lbc = (psi[i1, 0]**2).mean() + (psi[i2, 0]**2).mean()
        Ltot = Lpde + Lbc
        if tt == 0 or Ltot.detach().numpy() < Lbest:
            Lbest = Ltot.detach().numpy()
            best = [params.clone().detach() for params in params]
        if tt % 1 == 0:
            print("%8d: %.2e %.2e %.2e [%.5e]" %
                  (tt, Ltot.detach().numpy(), Lpde.detach().numpy(),
                   Lbc.detach().numpy(), Lbest))
        if tt == epochs:
            with torch.no_grad():
                for a, b in zip(params, best):
                    a.copy_(b)
            break
        tt += 1
        Ltot.backward(retain_graph=False)
        optimizer.step()


torch.manual_seed(12345)
dtype = torch.double
torch.set_default_tensor_type('torch.DoubleTensor')
bcutoff = 17.5
cutoff = 0.005
L = 18
n = 10000
Rlo = 0.2
Rhi = 3
sc_sampling = 1
nh = 16
ne = 32
nl = 10
H1a = ini(2, nh)
H1b = ini(nh)
H2a = ini(nh, nh)
H2b = ini(nh)
H3a = ini(nh, 1)
H3b = ini(1)
L1a = ini(1, nl)
L1b = ini(nl)
L2a = ini(nl, 1)
L2b = ini(1)
E1a = ini(1, ne)
E1b = ini(ne)
E2a = ini(ne, ne)
E2b = ini(ne)
E3a = ini(ne, 1)
E3b = ini(1)
x = torch.empty(n, 1, dtype=dtype, requires_grad=True)
y = torch.empty(n, 1, dtype=dtype, requires_grad=True)
z = torch.empty(n, 1, dtype=dtype, requires_grad=True)
R = torch.empty(n, 1, dtype=dtype)
params = (H1a, H1b, H2a, H2b, H3a, H3b, L1a, L1b, L2a, L2b, E1a, E1b, E2a, E2b,
          E3a, E3b)
train(params, lr=8e-3, epochs=1)
train((E1a, E1b, E2a, E2b, E3a, E3b), lr=1e-4, epochs=1)
with torch.no_grad():
    with open("model.bin", "wb") as file:
        for x in params:
            x = x.numpy()
            file.write(x.ndim.to_bytes(4, "little"))
            for d in x.shape:
                file.write(d.to_bytes(4, "little"))
            file.write(x.tobytes())
