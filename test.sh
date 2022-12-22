set -x
mkdir results

cd octave
octave-cli ./r_gmres.m > ../results/octave.csv
cd ..

for run in $(seq 1 5); do
    for i in 10 100 1000 10000; do
        build/gpuSam -n $i > results/normal_${i}_run$run.csv
    done
done

for run in $(seq 1 5); do
    for i in 10 100 200 500; do
        build/gpuSam -n $i -p > results/precond_${i}_run${run}.csv
    done
done

for run in $(seq 1 5); do
    for i in 10 100 200 500; do
        build/gpuSam -n $i -p -s > results/sam_${i}_run$run.csv
    done
done

