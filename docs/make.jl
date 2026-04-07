using Documenter, BASEforHANK

makedocs(;
    sitename = "Documentation for BASEforHANK module",
    pages = [
        "Home" => "index.md",
        "Household problem" => "HouseholdProblem.md",
        "Computational Notes" => "ComputationalNotes.md",
        "Example Structure" => "GeneralStructure.md",
        "Steady state" => "SteadyState.md",
        "Calibration" => "Calibration.md",
        "Perturbation solution" => "PerturbationSolution.md",
        "Estimation" => "Estimation.md",
        "IRF Matching" => "IRF_matching.md",
        "Post estimation" => "PostEstimation.md",
        "Utilities" => "Utilities.md",
    ],
    format = Documenter.HTML(;
        mathengine = Documenter.HTMLWriter.MathJax3(),
        prettyurls = true,
    ),
)

deploydocs(;
    repo = "github.com/BASEforHANK/BASEtoolbox.jl.git",
    versions = nothing,
    forcepush = true,
)
