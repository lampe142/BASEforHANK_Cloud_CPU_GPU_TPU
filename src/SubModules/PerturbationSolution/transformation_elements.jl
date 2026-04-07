"""
    transformation_elements(sr)

Prepare the elements required for compression and uncompression of variables
used in the linearized solution.

# Arguments

  - `sr`: steady-state structure (`SteadyResults`) containing steady-state values,
    indexes, numerical parameters, and other relevant data.

# Returns

  - An instance of `TransformationElements`.

The returned object contains transformation matrices and other elements
needed for compression and uncompression of variables, such as marginal
distributions and copula parameters.
"""
function transformation_elements(
    sr::SteadyResults,
    model::TwoAsset,
    distribution_states::CopulaStates,
)
    PDFSS = cdf_to_pdf(sr.distrSS.COP)
    # Matrices to take care of reduced degree of freedom in marginal distributions
    Γ = shuffleMatrix(PDFSS)
    # Matrices for discrete cosine transforms
    DC = Array{Array{Float64,2},1}(undef, 3)
    DC[1] = mydctmx(sr.n_par.nb)
    DC[2] = mydctmx(sr.n_par.nk)
    DC[3] = mydctmx(sr.n_par.nh)
    IDC = [DC[1]', DC[2]', DC[3]']

    DCD = Array{Array{Float64,2},1}(undef, 3)
    DCD[1] = mydctmx(sr.n_par.nb_copula)
    DCD[2] = mydctmx(sr.n_par.nk_copula)
    DCD[3] = mydctmx(sr.n_par.nh_copula)
    IDCD = [DCD[1]', DCD[2]', DCD[3]']

    # does not apply here
    pareto_indices = []

    return TransformationElements(Γ, DC, IDC, DCD, IDCD, pareto_indices)
end

function transformation_elements(
    sr::SteadyResults,
    model::OneAsset,
    distribution_states::CDFStates,
)
    PDF_ySS = cdf_to_pdf(sr.distrSS.CDF[end, :])
    # Matrices to take care of reduced degree of freedom in marginal distributions
    Γ = shuffleMatrix(PDF_ySS)
    # Matrices for discrete cosine transforms
    DC = Array{Array{Float64,2},1}(undef, 2)
    DC[1] = mydctmx(sr.n_par.nb)
    DC[2] = mydctmx(sr.n_par.nh)
    IDC = [DC[1]', DC[2]']

    DCD = Array{Array{Float64,2},1}(undef, 2)
    DCD[1] = mydctmx(sr.n_par.nb_copula)
    DCD[2] = mydctmx(sr.n_par.nh_copula)
    IDCD = [DCD[1]', DCD[2]']

    if (isa(sr.n_par.transf_CDF, ParetoTransformation)) && (isa(sr.distrSS, CDF))
        pareto_indices = set_pareto_indices(sr, sr.distrSS)
    else
        pareto_indices = []
    end

    return TransformationElements(Γ, DC, IDC, DCD, IDCD, pareto_indices)
end

function transformation_elements(
    sr::SteadyResults,
    model::OneAsset,
    distribution_states::CopulaStates,
)
    PDFSS = cdf_to_pdf(sr.distrSS.COP)
    # Matrices to take care of reduced degree of freedom in marginal distributions
    Γ = shuffleMatrix(PDFSS)
    # Matrices for discrete cosine transforms
    DC = Array{Array{Float64,2},1}(undef, 2)
    DC[1] = mydctmx(sr.n_par.nb)
    DC[2] = mydctmx(sr.n_par.nh)
    IDC = [DC[1]', DC[2]']

    DCD = Array{Array{Float64,2},1}(undef, 2)
    DCD[1] = mydctmx(sr.n_par.nb_copula)
    DCD[2] = mydctmx(sr.n_par.nh_copula)
    IDCD = [DCD[1]', DCD[2]']

    # does not apply here
    pareto_indices = []

    return TransformationElements(Γ, DC, IDC, DCD, IDCD, pareto_indices)
end

function transformation_elements(
    sr::SteadyResults,
    model::CompleteMarkets,
    distribution_states::Union{CDFStates,CopulaStates},
)
    PDF_ySS = cdf_to_pdf(sr.distrSS.h)
    # Matrices to take care of reduced degree of freedom in marginal distributions
    Γ = shuffleMatrix(PDF_ySS)
    # Matrices for discrete cosine transforms
    DC = Array{Array{Float64,2},1}(undef, 2)
    DC[1] = zeros(Float64, 1, 1)
    DC[2] = zeros(Float64, 1, 1)
    IDC = [DC[1]', DC[2]']

    DCD = Array{Array{Float64,2},1}(undef, 2)
    DCD[1] = zeros(Float64, 1, 1)
    DCD[2] = zeros(Float64, 1, 1)
    IDCD = [DCD[1]', DCD[2]']

    # does not apply here
    pareto_indices = []

    return TransformationElements(Γ, DC, IDC, DCD, IDCD, pareto_indices)
end

function set_pareto_indices(sr::SteadyResults, distrSS::CDF)
    CDFSS_cond = copy(distrSS.CDF)
    CDFSS_cond[:, 2:end] .= diff(CDFSS_cond; dims = 2)
    start_pareto = [
        findfirst(diff(CDFSS_cond[:, i_h]) .> sr.n_par.start_pareto_threshold) for
        i_h = 1:(sr.n_par.nh)
    ]
    start_pareto = [
        if !isnothing(idx)
            idx
        else
            1
        end for idx in start_pareto
    ]
    end_pareto = [
        findlast(diff(CDFSS_cond[:, i_h]) .> sr.n_par.end_pareto_threshold) for
        i_h = 1:(sr.n_par.nh)
    ]
    end_pareto = [
        if !isnothing(idx)
            idx + 1
        else
            sr.n_par.nb
        end for idx in end_pareto
    ]
    if sr.n_par.verbose
        println("Pareto transformation indices set to: $start_pareto to $end_pareto")
    end
    return [start_pareto, end_pareto]
end
