#usage: sh compile.sh filename.cpp

# # filename
# IN=${1}

# # your compiler


# # openmp flag of your compiler
OPENFLAG=-fopenmp
# #OPENFLAG=-openmp
# #OPENFLAG=-qopenmp

# # set the output name
# OUT=$(basename ${IN} .cpp).out

# ${CC} ${OPENFLAG} ${IN} -o ${OUT}

# ==========================
IN=${1}

# CC=clang++
CC=g++

OMP_PATH="/opt/homebrew/opt/libomp"

# FLAGS="-Xpreprocessor -fopenmp -I${OMP_PATH}/include -L${OMP_PATH}/lib -lomp"

FLAGS="-Xpreprocessor ${OPENFLAG} -I${OMP_PATH}/include -L${OMP_PATH}/lib -lomp"


OUT=$(basename ${IN} .cpp).out

${CC} ${FLAGS} ${IN} -o ${OUT}