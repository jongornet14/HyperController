for i in {0..9}
do
    python3 rl_tests.py --seed $i --env 'Reacher-v4' --total_frames 1000000 --t_ready 5 --method 'PB2' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'Reacher-v4' --total_frames 1000000 --t_ready 5 --method 'GP-UCB' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'Reacher-v4' --total_frames 1000000 --t_ready 5 --method 'Random' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'Reacher-v4' --total_frames 1000000 --t_ready 5 --method 'HyperController' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'Reacher-v4' --total_frames 1000000 --t_ready 5 --method 'HyperBand' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'Reacher-v4' --total_frames 1000000 --t_ready 5 --method 'Random_Start' --folder 'long-results'
    wait
done
wait 

for i in {0..9}
do
    python3 rl_tests.py --seed $i --env 'Pusher-v4' --total_frames 1000000 --t_ready 5 --method 'PB2' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'Pusher-v4' --total_frames 1000000 --t_ready 5 --method 'GP-UCB' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'Pusher-v4' --total_frames 1000000 --t_ready 5 --method 'Random' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'Pusher-v4' --total_frames 1000000 --t_ready 5 --method 'HyperController' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'Pusher-v4' --total_frames 1000000 --t_ready 5 --method 'HyperBand' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'Pusher-v4' --total_frames 1000000 --t_ready 5 --method 'Random_Start' --folder 'long-results'
    wait
done
wait 

for i in {0..9}
do
    python3 rl_tests.py --seed $i --env 'InvertedDoublePendulum-v4' --total_frames 1000000 --t_ready 5 --method 'PB2' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'InvertedDoublePendulum-v4' --total_frames 1000000 --t_ready 5 --method 'GP-UCB' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'InvertedDoublePendulum-v4' --total_frames 1000000 --t_ready 5 --method 'Random' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'InvertedDoublePendulum-v4' --total_frames 1000000 --t_ready 5 --method 'HyperController' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'InvertedDoublePendulum-v4' --total_frames 1000000 --t_ready 5 --method 'HyperBand' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'InvertedDoublePendulum-v4' --total_frames 1000000 --t_ready 5 --method 'Random_Start' --folder 'long-results'
    wait
done
wait 

for i in {0..9}
do
    python3 rl_tests.py --seed $i --env 'HalfCheetah-v4' --total_frames 1000000 --t_ready 5 --method 'PB2' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'HalfCheetah-v4' --total_frames 1000000 --t_ready 5 --method 'GP-UCB' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'HalfCheetah-v4' --total_frames 1000000 --t_ready 5 --method 'Random' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'HalfCheetah-v4' --total_frames 1000000 --t_ready 5 --method 'HyperController' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'HalfCheetah-v4' --total_frames 1000000 --t_ready 5 --method 'HyperBand' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'HalfCheetah-v4' --total_frames 1000000 --t_ready 5 --method 'Random_Start' --folder 'long-results'
    wait
done
wait 

for i in {0..9}
do
    python3 rl_tests.py --seed $i --env 'BipedalWalker-v3' --total_frames 1000000 --t_ready 5 --method 'PB2' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'BipedalWalker-v3' --total_frames 1000000 --t_ready 5 --method 'GP-UCB' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'BipedalWalker-v3' --total_frames 1000000 --t_ready 5 --method 'Random' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'BipedalWalker-v3' --total_frames 1000000 --t_ready 5 --method 'HyperController' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'BipedalWalker-v3' --total_frames 1000000 --t_ready 5 --method 'HyperBand' --folder 'long-results'
    wait
    python3 rl_tests.py --seed $i --env 'BipedalWalker-v3' --total_frames 1000000 --t_ready 5 --method 'Random_Start' --folder 'long-results'
    wait
done
wait 