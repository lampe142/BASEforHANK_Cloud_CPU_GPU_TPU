"""
    plot_distributional_irfs(
        shocks_to_plot::Vector{Tuple{Symbol,String}},
        vars_to_plot::Vector{Tuple{String,String}},
        IRFs_to_plot::Dict{String,Array{Float64}},
        IRFs_order::Vector{Symbol},
        n_par;
        bounds::Dict{String,Tuple{Float64,Float64}} = Dict{String,Tuple{Float64,Float64}}(),
        horizon::Int64 = 40,
        legend::Bool = false,
        show_fig::Bool = true,
        save_fig::Bool = false,
        path::String = "",
        fps::Int = 2,
        suffix::String = ""
    )

Plots distributional impulse response functions (IRFs) for specified shocks and variables.

This function generates either static 3D surface plots for univariate distributions or
animated 3D surface plots for bivariate distributions. It iterates through the specified
shocks and variables, creating and optionally saving a plot for each combination.

# Arguments

  - `shocks_to_plot::Vector{Tuple{Symbol,String}}`: A vector of tuples, where each tuple
    contains a shock variable (as a `Symbol`) and its corresponding label (`String`).
  - `vars_to_plot::Vector{Tuple{Symbol,String}}`: A vector of tuples, each containing a
    variable to plot (as a `Symbol`) and its corresponding label (`String`).
  - `IRFs_to_plot::Dict{String,Array{Float64}}`: A dictionary of distributional IRFs, where
    the keys are variable names and the values are arrays of IRFs.
  - `IRFs_order::Vector{Symbol}`: A vector of symbols specifying the order of shocks in the
    IRF arrays.
  - `n_par`: Model parameters, needed for distributional IRFs. Defaults to nothing.

# Keyword Arguments

  - `bounds::Dict{String,Tuple{Float64,Float64}}`: A dictionary to set axis limits for
    specific variables. Keys can be "b", "k", "h". Example: `Dict("k" => (0.0, 50.0))`.
  - `horizon::Int64`: The time horizon (number of periods) over which IRFs are plotted.
  - `legend::Bool`: Toggles the display of the legend on plots. Default is `false`.
  - `show_fig::Bool`: If `true`, displays the plot (default: `true`).
  - `save_fig::Bool`: If `true`, saves the plot as a PDF. Default is `false`.
  - `path::String`: The directory path where the generated plots should be saved. Default is
    an empty string (no saving).
  - `fps::Int`: Frames per second for animated GIF outputs. Default is `2`.
  - `suffix::String`: A suffix to append to the saved plot filenames. Default is an empty
    string.
"""
function plot_distributional_irfs(
    shocks_to_plot::Vector{Tuple{Symbol,String}},
    vars_to_plot::Vector{Tuple{String,String}},
    IRFs_to_plot::Dict{String,Array{Float64}},
    IRFs_order::Vector{Symbol},
    n_par;
    bounds::Dict{String,Tuple{Float64,Float64}} = Dict{String,Tuple{Float64,Float64}}(),
    horizon::Int64 = 40,
    legend::Bool = false,
    show_fig::Bool = true,
    save_fig::Bool = false,
    path::String = "",
    fps::Int = 2,
    suffix::String = "",
)

    # Remove variables from vars_to_plot that are not in IRFs_to_plot
    filtered_vars_to_plot = []
    for (var, lab) in vars_to_plot
        if var in keys(IRFs_to_plot)
            push!(filtered_vars_to_plot, (var, lab))
        else
            @warn "The variable $var not found in IRFs_to_plot, removed from vars_to_plot."
        end
    end
    vars_to_plot = filtered_vars_to_plot

    # Remove variables from vars_to_plot that have unsupported number of dimensions
    filtered_vars_to_plot = []
    for (var, lab) in vars_to_plot
        if ndims(IRFs_to_plot[var]) in (3, 4)
            push!(filtered_vars_to_plot, (var, lab))
        else
            @warn "The variable $var has unsupported number of dimensions $(ndims(IRFs_to_plot[var])), removed from vars_to_plot."
        end
    end
    vars_to_plot = filtered_vars_to_plot

    # Unpack variables (fields) and labels
    vars = [vars_to_plot[i][1] for i in eachindex(vars_to_plot)]
    labs = [vars_to_plot[i][2] for i in eachindex(vars_to_plot)]

    # Remove shocks from shocks_to_plot that are not in IRFs_order
    filtered_shocks_to_plot = []
    for (i_shock, i_shock_lab) in shocks_to_plot
        if i_shock in IRFs_order
            push!(filtered_shocks_to_plot, (i_shock, i_shock_lab))
        else
            @warn "The shock $i_shock not found in IRFs_order, removed from shocks_to_plot."
        end
    end
    shocks_to_plot = filtered_shocks_to_plot

    # Create plots for each shock
    for (i_shock, i_shock_lab) in shocks_to_plot

        # Find position of current shock (i_shock) in IRFs array (IRFs_order)
        idx = findfirst(x -> x == i_shock, IRFs_order)

        # Create a plot for each variable
        for (i, (var, lab)) in enumerate(zip(vars, labs))
            # Select IRFs for the variable
            i_IRFs = IRFs_to_plot[var]

            if ndims(i_IRFs) == 3
                i_IRFs = i_IRFs[:, 1:horizon, idx]

                p = plot_univariate_plot(
                    i_IRFs,
                    var,
                    lab,
                    i_shock_lab,
                    horizon,
                    bounds,
                    n_par;
                    legend = false,
                )

                # Save plot
                if save_fig
                    savefig(
                        p,
                        path * "/DistIRFs_" * string(i_shock) * "_" * var * suffix * ".pdf",
                    )
                end

                # Show plot
                if show_fig
                    display(p)
                end

            elseif ndims(i_IRFs) == 4

                # Select shock and horizon
                i_IRFs = i_IRFs[:, :, 1:horizon, idx]

                # Create plot
                anim = plot_bivariate_animation(
                    i_IRFs[:, :, :],
                    var,
                    lab,
                    i_shock_lab,
                    horizon,
                    bounds,
                    n_par;
                    legend = legend,
                )

                # Save plot
                if save_fig
                    anim = gif(
                        anim,
                        path * "/DistIRFs_" * string(i_shock) * "_" * var * suffix * ".gif";
                        fps = fps,
                        show_msg = false,
                    )
                else
                    anim = gif(anim; fps = fps, show_msg = false)
                end

                # Show plot
                if show_fig
                    display(anim)
                end
            end
        end
    end
end

"""
    infer_grid_label_and_object(var::String, n_par) -> Tuple

A helper function that parses a variable name string to determine the corresponding grid(s),
axis label(s), and the type of quantity being plotted (Density or Marginal Value).

It assumes a specific variable naming convention, such as `Wk_k` (Marginal Value `W` of `k`
over state `k`) or `D_b` (Density `D` over state `b`).

# Arguments

  - `var::String`: The variable name string (e.g., "Wk_k", "D_kh").
  - `n_par`: A parameters object containing the model's grids (`.grid_b`, `.grid_k`, etc.).

# Returns

  - A `Tuple` containing:

     1. Grid(s): A single grid vector or a tuple of two grid vectors.
     2. Label(s): A string label for the axis or a tuple of two labels.
     3. Object type: A string, either "Density" or "Marginal Value".
"""
function infer_grid_label_and_object(var, n_par)
    var = strip(var)
    measure = split(var, "_")[end]
    density_or_mv = "$(split(var, "_")[1][1])" == "W" ? "Marginal Value" : "Density"

    if measure == "b"
        return n_par.grid_b, "Bonds", density_or_mv
    elseif measure == "k"
        return n_par.grid_k, "Capital", density_or_mv
    elseif measure == "h"
        return n_par.grid_h, "Human Capital", density_or_mv
    elseif measure == "bk"
        return (n_par.grid_b, n_par.grid_k), ("Bonds", "Capital"), density_or_mv
    elseif measure == "bh"
        return (n_par.grid_b, n_par.grid_h), ("Bonds", "Human Capital"), density_or_mv
    elseif measure == "kh"
        return (n_par.grid_k, n_par.grid_h), ("Capital", "Human Capital"), density_or_mv
    else
        error("Unknown measure: $measure")
    end
end

"""
    plot_bivariate_animation(
        data::Array{Float64,3},
        var::String,
        lab::String,
        i_shock_lab::String,
        horizon::Int,
        bounds::Dict,
        n_par;
        legend::Bool = false
    )

Creates a 3D animated surface plot for a bivariate distribution evolving over time.

# Arguments

  - `data::Array{Float64,3}`: A 3D array (`dim1` x `dim2` x `time`) of the bivariate IRF.
  - `var::String`: The internal variable name (e.g., "D_kh").
  - `lab::String`: The descriptive label for the plot title (e.g., "Distribution over
    Capital and Human Capital").
  - `i_shock_lab::String`: The label for the shock causing the response.
  - `horizon::Int`: The number of periods to animate.
  - `bounds::Dict`: The dictionary containing axis limit specifications.
  - `n_par`: The parameters object containing model grids.

# Keyword Arguments

  - `legend::Bool`: Toggles the plot legend. Default is `false`.

# Returns

  - `Animation`: A `Plots.Animation` object that can be displayed or saved as a GIF.
"""
function plot_bivariate_animation(
    data::Array{Float64,3},
    var,
    lab,
    i_shock_lab,
    horizon,
    bounds,
    n_par;
    legend = false,
)
    dist_grid, lab_grid, density_or_mv = infer_grid_label_and_object(var, n_par)

    # Define a mapping from the axis label to the corresponding key in the `bounds` dict
    bounds_key_map = Dict("Bonds" => "b", "Capital" => "k", "Human Capital" => "h")

    # Determine the x-limits based on the first axis label
    x_label = lab_grid[1]
    x_bounds_key = get(bounds_key_map, x_label, "") # Safely get the key
    x_limits = haskey(bounds, x_bounds_key) ? bounds[x_bounds_key] : :auto

    # Determine the y-limits based on the second axis label
    y_label = lab_grid[2]
    y_bounds_key = get(bounds_key_map, y_label, "") # Safely get the key
    y_limits = haskey(bounds, y_bounds_key) ? bounds[y_bounds_key] : :auto

    anim = @animate for i ∈ 1:horizon
        Plots.surface(
            dist_grid[1],
            dist_grid[2],
            data[:, :, i]';
            camera = (70, 30),
            size = (600, 500),
            xlabel = "\n" * lab_grid[1],
            ylabel = "\n" * lab_grid[2],
            zlabel = density_or_mv,
            xlims = x_limits,
            ylims = y_limits,
            color = :winter,
            legend = legend,
            title = "$lab for shock to $i_shock_lab,\n Horizon: $i",
            titlefontsize = 10,
            display_option = Plots.GR.OPTION_SHADED_MESH,
        )
    end

    return anim
end

"""
    plot_univariate_plot(
        data,
        var::String,
        lab::String,
        i_shock_lab::String,
        horizon::Int,
        bounds::Dict,
        n_par;
        legend::Bool = false
    )

Creates a 3D static surface plot for a univariate distribution evolving over time.

The x-axis represents the state variable, the y-axis represents time (horizon), and the
z-axis represents the density or marginal value. This function filters the data based on the
provided `bounds` *before* plotting to ensure the z-axis is scaled correctly to the visible
data.

# Arguments

  - `data::Array{Float64,2}`: A 2D array (`grid_points` x `time`) of the univariate IRF.
  - `var::String`: The internal variable name (e.g., "Wk_k").
  - `lab::String`: The descriptive label for the plot title (e.g., "Marginal Value of
    Capital").
  - `i_shock_lab::String`: The label for the shock causing the response.
  - `horizon::Int`: The number of time periods to plot on the y-axis.
  - `bounds::Dict`: The dictionary containing axis limit specifications (e.g., `Dict("k" => (0.0, 50.0))`).
  - `n_par`: The parameters object containing model grids.

# Keyword Arguments

  - `legend::Bool`: Toggles the plot legend. Default is `false`.

# Returns

  - `Plots.Plot`: A plot object that can be displayed or saved.
"""
function plot_univariate_plot(
    data,
    var,
    lab,
    i_shock_lab,
    horizon,
    bounds,
    n_par;
    legend = false,
)
    # Infer grid, object
    dist_grid, lab_grid, density_or_mv = infer_grid_label_and_object(var, n_par)

    # Define a mapping from the axis label to the corresponding key in the `bounds` dict
    bounds_key_map = Dict("Bonds" => "b", "Capital" => "k", "Human Capital" => "h")

    # Determine the x-limits based on the first axis label
    x_bounds_key = get(bounds_key_map, lab_grid, "")
    x_limits =
        haskey(bounds, x_bounds_key) ? bounds[x_bounds_key] :
        (first(dist_grid), last(dist_grid))

    # Find the indices of the grid points that are within our desired x-limits
    visible_indices = findall(x -> x_limits[1] <= x <= x_limits[2], dist_grid)

    # Filter both the grid and the data to only include these points
    filtered_grid = dist_grid[visible_indices]
    filtered_data = data[visible_indices, :]

    # Create the plot using the FILTERED data and grid
    p = Plots.surface(
        filtered_grid,
        1:horizon,
        filtered_data[:, :]';
        camera = (60, 50),
        size = (600, 500),
        color = :winter,
        legend = legend,
        ylabel = "Horizon",
        xlabel = lab_grid,
        zlabel = density_or_mv,
        display_option = Plots.GR.OPTION_SHADED_MESH,
    )

    # fontsize for title
    fs = density_or_mv == "Marginal Value" ? 10 : 12

    plot!(p; title = lab * " for shock to " * i_shock_lab, titlefontsize = fs)

    return p
end
