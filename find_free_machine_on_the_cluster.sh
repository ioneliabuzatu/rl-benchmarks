username=$1

echo $username

declare -a servers=("cayman" "cheetah" "cobra" "cougar" "elk" "fox" "grizzly" "lizard" "lynx" "orca" "panther" "pike" "raptor" "scorpion" "shark" "spider" "viper" "wasp" "wolf")

for server in ${servers[@]}
do
	echo @$server:
	ssh buzatu@$server.ml.jku.at -p 5792 nvidia-smi --format=csv --query-gpu=name,index,utilization.gpu,memory.used,memory.total
	echo 
done


