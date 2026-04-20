println("Julia Version Running: ", VERSION)
using Pkg
Pkg.activate(".")
#Pkg.add("PrettyTables")
#Pkg.status(["DataFrames", "PrettyTables"])
println("Number of threads Julia is using: ", Threads.nthreads())