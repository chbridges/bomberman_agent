while true
do
echo
date
printf "rise:   "
cat ./logs/rise_0.log | grep "ROUND"  | tail -1 | awk '{print $NF}'
sleep 5
done
