#usage: sh compile.sh filename.cpp

CC='mpicxx'
IN=${1}  # ${1} means input filename 
OUT=$(basename ${IN} .cpp).out

# compiler
${CC} ${IN} -o ${OUT}

# execute
mpirun -np 12 ./${OUT}
