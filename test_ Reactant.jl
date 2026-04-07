using Random, Reactant

# 1. Set up an explicit RNG and generate data on the host
rng = MersenneTwister(1234)
x_host = randn(rng, Float32, 1024)  # reproducible host-side randomness

# 2. Move data to Reactant
x_ra = Reactant.ConcreteRArray(x_host)

# 3. Define a *deterministic* kernel for Reactant
function scale!(x, α)
    @. x = α * x
    return x
end

# 4. Compile and run with Reactant
f = Reactant.compile(scale!, (x_ra, 0.1f0))
f(x_ra, 0.1f0)