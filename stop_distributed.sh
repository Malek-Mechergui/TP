HOSTS=("austin" "baton-rouge" "hartford" "oklahoma-city")

for host in ${HOSTS[@]}; do
   ssh $host "(pkill -9 -u `whoami` 'python|torchrun')&" &
done
