: 'run_ms.sh'
IMGDIR='./HW2_ImageData/Images'
OUTDIR='./out_ms'

for sp in 2 8 32 128
do
	for sr in 2 8 32 128
	do
		python3 mean_shift.py --sp=$sp --sr=$sr --imgdir=$IMGDIR --outdir=$OUTDIR
	done
done