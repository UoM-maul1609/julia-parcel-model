import Pkg; Pkg.add("YAML")
import YAML
data = YAML.load_file("namelist.yml")
println(data)
