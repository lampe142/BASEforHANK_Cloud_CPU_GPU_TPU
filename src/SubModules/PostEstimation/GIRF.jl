"""
    GIRF_FO(l, ν, lr, sr; shock_names=shock_names, η=nothing)

Compute the first-order Generalized Impulse Response Function (GIRF)
up to horizon `l` for a given error vector `ν`.
"""
function GIRF_FO(l, ν, lr, sr, shock_names = shock_names, η = nothing)
    if isnothing(η)
        η = sparse(
            [getfield(sr.indexes, s) for s in shock_names],
            collect(1:length(shock_names)),
            ones(length(shock_names)),
            sr.n_par.nstates,
            length(shock_names),
        )
    end
    irf_arr = Vector{Union{Nothing,irf}}(nothing, l)
    irf_arr[1] = irf(η * ν, lr.State2Control * η * ν)
    for i = 2:l
        irf_x = lr.LOMstate * irf_arr[i - 1].x
        irf_arr[i] = irf(irf_x, lr.State2Control * irf_x)
    end
    return irf_arr
end

"""
    GIRF_SO(l, ν, lr_full, lr_reduc, sr_reduc, sor_full, variances, xf_reduc; shock_names=shock_names, η_red=nothing, η_full=nothing)

Compute the second-order Generalized Impulse Response Function (GIRF)
at horizon `l` for a given error vector `ν`.
"""
function GIRF_SO(
    l,
    ν,
    lr_full,
    lr_reduc,
    sr_reduc,
    sor_full,
    variances,
    xf_reduc,
    shock_names = shock_names,
    η_red = nothing,
    η_full = nothing,
)
    if isnothing(η_red)
        η_red = sparse(
            [getfield(sr_reduc.indexes_r, s) for s in shock_names],
            collect(1:length(variances)),
            ones(length(variances)),
            sr_reduc.n_par.nstates_r,
            length(variances),
        )
    end
    ηsd = η_red * Diagonal(sqrt.(variances))

    nozero(x) = x == 0 ? 1.0e-8 : x # just auxiliary function
    S = Diagonal(ν ./ nozero.(ν))
    Λ = ((ηsd * (I - S)) ⊗ (ηsd * (I - S)) - (ηsd ⊗ ηsd)) * vec(I(length(variances)))

    # need first order irf
    irf_fo = GIRF_FO(l, ν, lr_full, sr_reduc, shock_names, η_full)

    # Preallocate arrays
    irf_arr = Vector{Union{Nothing,irf}}(nothing, l)
    girf_kron_g_arr = Vector{Union{Nothing,Matrix{Float64}}}(nothing, l)
    x_s_sum = zeros(size(lr_full.LOMstate, 1))
    irf_x = zeros(size(lr_full.LOMstate, 1))
    term_kron = zeros(size(lr_full.LOMstate, 1)^2, 1)

    LOMstate0 = zeros(Float64, size(lr_reduc.LOMstate)) + I
    LOMstate1 = lr_reduc.LOMstate * LOMstate0

    function girf_kron!(term, LOMstate0, LOMstate1)
        LOMstateην = LOMstate0 * η_red * ν
        term .=
            ((LOMstate1 * xf_reduc) ⊗ LOMstateην) +
            (LOMstateην ⊗ (LOMstate1 * xf_reduc)) +
            (LOMstateην ⊗ LOMstateην) +
            ((LOMstate0 ⊗ LOMstate0)) * Λ
    end

    girf_kron!(term_kron, LOMstate0, LOMstate1)
    girf_kron_g_arr[1] = sor_full.gxx * term_kron
    irf_y = lr_full.State2Control * irf_fo[1].x + 0.5 * girf_kron_g_arr[1]
    irf_arr[1] = irf(irf_fo[1].x, irf_y)
    if sr_reduc.n_par.verbose
        println("period 1 done")
    end
    for i = 2:l
        timestart = now()
        x_s_sum .= lr_full.LOMstate * x_s_sum + 0.5 * sor_full.hxx * term_kron
        irf_x .= x_s_sum + irf_fo[i].x

        LOMstate0 .= LOMstate1
        LOMstate1 .= lr_reduc.LOMstate * LOMstate0

        girf_kron!(term_kron, LOMstate0, LOMstate1)
        girf_kron_g_arr[i] = sor_full.gxx * term_kron
        irf_y = lr_full.State2Control * irf_x + 0.5 * girf_kron_g_arr[i]
        irf_arr[i] = irf(irf_x, irf_y)
        if sr_reduc.n_par.verbose
            println("period ", i, " done")
        end
        timer_help(timestart)
    end
    return irf_arr, girf_kron_g_arr
end

"""
    GIRF_TO(l, ν, lr_full, lr_reduc, sr_reduc, sor_full, sor_reduc, tor_full, variances, xf_reduc, xs_reduc, Eε3; shock_names=shock_names)

Compute the third-order Generalized Impulse Response Function (GIRF)
at horizon `l` for a given error vector `ν`.
"""
function GIRF_TO(
    l,
    ν,
    lr_full,
    lr_reduc,
    sr_reduc,
    sor_full,
    sor_reduc,
    tor_full,
    variances,
    xf_reduc,
    xs_reduc,
    Eε3,
    shock_names = shock_names,
)
    η = sparse(
        [getfield(sr_reduc.indexes_r, s) for s in shock_names],
        collect(1:length(variances)),
        ones(length(variances)),
        sr_reduc.n_par.nstates_r,
        length(variances),
    )
    ηsd = η * Diagonal(sqrt.(variances))

    nozero(x) = x == 0 ? 1.0e-8 : x # just auxiliary function
    S = Diagonal(ν ./ nozero.(ν))
    Λ = (kron(ηsd * (I - S), ηsd * (I - S)) - kron(ηsd, ηsd)) * vec(I(length(variances)))

    sum_z(arr, n) = !isempty(arr) ? sum(arr) : zeros(n)

    hjms = [lr_reduc.LOMstate^k for k = 0:l]
    girf_kron3(j) =
        kron(
            hjms[j + 1] * xf_reduc,
            hjms[j] * η * ν,
            lr_reduc.LOMstate * hjms[j] * xf_reduc,
        ) +
        kron(hjms[j] * η * ν, (kron(hjms[j + 1], hjms[j + 1]) * kron(xf_reduc, xf_reduc))) +
        kron((kron(hjms[j + 1], hjms[j + 1]) * kron(xf_reduc, xf_reduc)), hjms[j] * η * ν) +
        kron(kron(hjms[j], hjms[j]) * (Λ + kron(η * ν, η * ν)), hjms[j + 1] * xf_reduc) +
        kron(hjms[j + 1] * xf_reduc, kron(hjms[j], hjms[j]) * (Λ + kron(η * ν, η * ν))) +
        kron(hjms[j] * η * ν, hjms[j + 1] * xf_reduc, hjms[j] * η * ν) +
        (
            kron(hjms[j] * ηsd * (I - S), hjms[j + 1] * xf_reduc, hjms[j] * ηsd * (I - S)) -
            kron(hjms[j] * ηsd, hjms[j + 1] * xf_reduc, hjms[j] * ηsd)
        ) * vec(I(length(variances))) +
        kron(hjms[j] * η * ν, hjms[j] * ηsd * (I - S), hjms[j] * ηsd * (I - S)) *
        vec(I(length(variances))) +
        kron(hjms[j] * ηsd * (I - S), hjms[j] * ηsd * (I - S), hjms[j] * η * ν) *
        vec(I(length(variances))) +
        kron(hjms[j] * ηsd * (I - S), hjms[j] * η * ν, hjms[j] * ηsd * (I - S)) *
        vec(I(length(variances))) +
        (
            kron(
                hjms[j] * ηsd * (I - S),
                hjms[j] * ηsd * (I - S),
                hjms[j] * ηsd * (I - S),
            ) - kron(hjms[j] * ηsd, hjms[j] * ηsd, hjms[j] * ηsd)
        ) * Eε3 +
        sum_z(
            [
                kron(
                    hjms[j] * η * ν,
                    kron(hjms[j - k + 1] * ηsd, hjms[j - k + 1] * ηsd) *
                    vec(I(length(variances))),
                ) +
                kron(hjms[j - k + 1] * ηsd, hjms[j] * η * ν, hjms[j - k + 1] * ηsd) *
                vec(I(length(variances))) +
                kron(
                    kron(hjms[j - k + 1] * ηsd, hjms[j - k + 1] * ηsd) *
                    vec(I(length(variances))),
                    hjms[j] * η * ν,
                ) for k = 2:j
            ],
            sr_reduc.n_par.nstates_r^3,
        )

    kronhx_r = kron(lr_reduc.LOMstate, lr_reduc.LOMstate)
    sr_r = sr_reduc
    @set! sr_r.indexes = sr_r.indexes_r
    @set! sr_r.n_par.nstates = sr_r.n_par.nstates_r
    irf_fo_red = GIRF_FO(l, ν, lr_reduc, sr_r, shock_names)

    girf_fs_arr =
        [Vector{Union{Nothing,Float64}}(nothing, sr_reduc.n_par.nstates_r^2) for _ = 1:l]
    fs_sum = zeros(sr_reduc.n_par.nstates_r^2)
    ψ = kron(
        η * ν,
        (
            lr_reduc.LOMstate * xs_reduc +
            0.5 * (sor_reduc.hxx * kron(xf_reduc, xf_reduc) + sor_reduc.hσσ)
        ),
    )
    girf_fs_arr[1] = ψ
    if sr_reduc.n_par.verbose
        println("prepare period 1 done")
    end
    for i = 2:l
        timestart = now()
        ψ = kronhx_r * ψ
        fs_sum =
            kronhx_r * fs_sum +
            kron(lr_reduc.LOMstate, 0.5 * sor_reduc.hxx) * girf_kron3(i - 1) +
            kron(lr_reduc.LOMstate, 0.5 * sor_reduc.hσσ) * irf_fo_red[i - 1].x
        girf_fs_arr[i] = (fs_sum + ψ)[:]
        if sr_reduc.n_par.verbose
            println("prepare period ", i, " done")
        end
        timer_help(timestart)
    end
    irf_so, girf_kron_g =
        GIRF_SO(l, ν, lr_full, lr_reduc, sr_r, sor_full, variances, xf_reduc, shock_names)

    irf_arr = Vector{Union{Nothing,irf}}(nothing, l)
    x_rd_sum = zeros(sr_r.n_par.nstates)
    irf_arr[1] = irf(
        irf_so[1].x,
        lr_full.State2Control * irf_so[1].x +
        0.5 * girf_kron_g[1] +
        sor_full.gxx * girf_fs_arr[1] +
        tor_full.gxxx * girf_kron3(1) / 6 +
        3 * tor_full.gxσσ * irf_fo_red[1].x / 6,
    )
    if sr_reduc.n_par.verbose
        println("period 1 done")
    end
    for i = 2:l
        timestart = now()
        x_rd_sum =
            lr_full.LOMstate * x_rd_sum +
            sor_full.hxx * girf_fs_arr[i - 1] +
            tor_full.hxxx * girf_kron3(i - 1) / 6 +
            3 * tor_full.hxσσ * irf_fo_red[i - 1].x / 6
        irf_x = x_rd_sum + irf_so[i].x
        irf_arr[i] = irf(
            irf_x,
            lr_full.State2Control * irf_x +
            0.5 * girf_kron_g[i] +
            sor_full.gxx * girf_fs_arr[i] +
            tor_full.gxxx * girf_kron3(i) / 6 +
            3 * tor_full.gxσσ * irf_fo_red[i].x / 6,
        )
        if sr_reduc.n_par.verbose
            println("period ", i, " done")
        end
        timer_help(timestart)
    end
    return irf_arr
end

"""
    uncond_IRF_SO(l, ν, lr_full, lr_reduc, sr_reduc, sor_full, variances, N_samp, x_f; shock_names=shock_names, ratio=0.01)

Compute the unconditional second-order Generalized Impulse Response Function (GIRF)
using Monte Carlo integration.
"""
function uncond_IRF_SO(
    l,
    ν,
    lr_full,
    lr_reduc,
    sr_reduc,
    sor_full,
    variances,
    N_samp,
    x_f,
    shock_names = shock_names;
    ratio = 0.01,
)
    println("Monte Carlo integration for unconditional second order IRF...")
    timer = time()
    irf_zero = irf(zeros(sr_reduc.n_par.nstates), zeros(sr_reduc.n_par.ncontrols))
    sum_irf = [irf_zero for i = 1:l]
    inds = [rand(1:N_samp) for i = 1:Int(N_samp * ratio)]
    irf_sos = [sum_irf for i = 1:Int(N_samp * ratio)]
    Threads.@threads for j = 1:Int(N_samp * ratio)
        #global inds, irf_sos
        ind = inds[j]
        irf_sos[j] = GIRF_SO(
            l,
            ν,
            lr_full,
            lr_reduc,
            sr_reduc,
            sor_full,
            variances,
            x_f[:, ind],
            shock_names,
        )[1]
    end
    for j = 1:Int(N_samp * ratio)
        #global sum_irf
        for k = 1:l
            @set! sum_irf[k].x = sum_irf[k].x + irf_sos[j][k].x
            @set! sum_irf[k].y = sum_irf[k].y + irf_sos[j][k].y
        end
    end
    # take average for *unconditional irf*
    irf_so = Vector{Union{Nothing,irf}}(nothing, l)
    for k = 1:l
        #global irf_so, sum_irf
        irf_so[k] = irf(sum_irf[k].x / (N_samp * ratio), sum_irf[k].y / (N_samp * ratio))
    end
    println("Monte Carlo integration done, in ", time() - timer, " seconds")
    return irf_so
end

"""
    uncond_IRF_TO(l, ν, lr_full, lr_reduc, sr_reduc, sor_full, sor_reduc, tor_full, variances, Eε3, N_samp, xfxs; shock_names=shock_names, ratio=0.01)

Compute the unconditional third-order Generalized Impulse Response Function (GIRF)
using Monte Carlo integration.
"""
function uncond_IRF_TO(
    l,
    ν,
    lr_full,
    lr_reduc,
    sr_reduc,
    sor_full,
    sor_reduc,
    tor_full,
    variances,
    Eε3,
    N_samp,
    xfxs,
    shock_names = shock_names;
    ratio = 0.01,
)
    println("Monte Carlo integration for unconditional third order IRF...")
    timer = time()
    irf_zero = irf(zeros(sr_reduc.n_par.nstates), zeros(sr_reduc.n_par.ncontrols))
    sum_irf = [irf_zero for i = 1:l]
    inds = [rand(1:N_samp) for i = 1:Int(N_samp * ratio)]
    irf_tos = [sum_irf for i = 1:Int(N_samp * ratio)]
    Threads.@threads for j = 1:Int(N_samp * ratio)
        #global inds, irf_tos
        ind = inds[j]
        x_f = xfxs[1:(sr_reduc.n_par.nstates_r), ind]
        x_s = xfxs[(sr_reduc.n_par.nstates_r + 1):(2 * sr_reduc.n_par.nstates_r), ind]
        irf_tos[j] = GIRF_TO(
            l,
            ν,
            lr_full,
            lr_reduc,
            sr_reduc,
            sor_full,
            sor_reduc,
            tor_full,
            variances,
            x_f,
            x_s,
            Eε3,
            shock_names,
        )
    end
    for j = 1:Int(N_samp * ratio)
        #global sum_irf
        for k = 1:l
            @set! sum_irf[k].x = sum_irf[k].x + irf_tos[j][k].x
            @set! sum_irf[k].y = sum_irf[k].y + irf_tos[j][k].y
        end
    end
    # take average for *unconditional irf*
    irf_to = Vector{Union{Nothing,irf}}(nothing, l)
    for k = 1:l
        #global irf_to, sum_irf
        irf_to[k] = irf(sum_irf[k].x / (N_samp * ratio), sum_irf[k].y / (N_samp * ratio))
    end
    println("Monte Carlo integration done, in ", time() - timer, " seconds")
    return irf_to
end
