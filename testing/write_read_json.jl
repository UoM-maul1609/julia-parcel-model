import Pkg; Pkg.add("JSON")
Pkg.add("DataStructures")
import JSON
import DataStructures
###################
### Write data ####
###################
# dictionary to write
dict1 = Dict("param1" => 1, "param2" => 2,
            "dict" => Dict("d1"=>1.,"d2"=>1.,"d3"=>1.))
            
# pass data as a json string (how it shall be displayed in a file)
stringdata = JSON.json(dict1)

# write the file with the stringdata variable information
open("write_read.json", "w") do f
        write(f, stringdata)
     end

###################
### Read data #####
###################
json_string = read("write_read.json", String)
dict2=JSON.parse(json_string,dicttype=Dict,inttype=Int64)

# print both dictionaries
println(dict1)
println(dict2)



# read some other data

json_string = read("example.json", String)
dict3=JSON.parse(json_string,dicttype=Dict,inttype=Int64)

# print the dictionary
println(dict3)