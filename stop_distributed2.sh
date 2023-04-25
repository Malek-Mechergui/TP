#HOSTS=("austin" "baton-rouge" "hartford" "oklahoma-city")
HOSTS=("jefferson-city" "lansing" "lincoln" "little-rock")

for host in ${HOSTS[@]}; do
   ssh $host "(pkill -9 -u `whoami` 'python|torchrun')&" &
done
