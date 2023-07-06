import Pkg; Pkg.add("NetCDF")

using NetCDF

# https://github.com/JuliaGeo/NetCDF.jl
filename = "myfile.nc"
varname  = "var1"
attribs  = Dict("units"   => "mm/d",
                "data_min" => 0.0,
                "data_max" => 87.0)

nc=nccreate(filename, varname, "x1", collect(11:20), "t", 20, Dict("units"=>"s"), atts=attribs)


# This will create the variable called var1 in the file myfile.nc. 
# The attributes defined in the Dict attribs are written to the file and are associated 
# with the newly created variable. The dimensions "x1" and "t" of the variable 
# are called "x1" and "t" in this example. If the dimensions do not exist yet in the file, 
# they will be created. The dimension "x1" will be of length 10 and have the values 11..20, 
# and the dimension "t" will have length 20 and the attribute "units" with the value "s".

# Now we can write data to the file:

d = rand(10, 20)
ncwrite(d, filename, varname)

#NetCDF.sync(nc)

# another example
fn = "myfile2.nc"
x=[0.1*i for i=1:100];nx=length(x)
y=[0.1*i for i=1:200];ny=length(y)

xdim = NcDim("x",nx,values=x)
ydim = NcDim("y",ny,values=y)
tdim = NcDim("t",0,unlimited=true)

uvar = NcVar("u",[xdim,ydim,tdim],t=Float32)
tvar = NcVar("t",tdim,t=Int64)
tvar.atts = Dict("units" => "seconds")

# fn=tempname()
ncu = NetCDF.create(fn,[uvar,tvar],mode=NC_NETCDF4)

for i=1:10
  u=rand(nx,ny)
  NetCDF.putvar(ncu,"u",Float32.(u),start=[1,1,i],count=[-1,-1,1])
  NetCDF.putvar(ncu,"t",Int64[i*5],start=[i]) #Write the time
end
NetCDF.sync(ncu)