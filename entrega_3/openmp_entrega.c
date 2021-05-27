// TODO: calculo del tiempo como la gente, borrar los calculos te tiempo de la entrega anterior

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define PI 3.14159265358979323846
#define DOUBLE_PI PI * 2

/*****************************************************************/

// Para calcular tiempo - Función de la cátedra
double dwalltime(){
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec / 1000000.0;
	return sec;
}

// Para generar un número aleatorio - Función de la cátedra
double randFP(double min, double max) {
	double range = (max - min);
	double div = RAND_MAX / range;
	return min + (rand() / div);
}

/*****************************************************************/

int COORDINATOR = 0;

// Main del programa
int main(int argc, char* argv[]){

	double *A, *B, *C, *C_secuencial, *T, *M, *R1, *R2, *RA, *RB, num, aSin, aCos, timetick_start, timetick_end, *ablk, *bblk, *cblk, average1;
	int N, i, j, k, bs, offset_i, offset_j, row_index, f, c, h, offset_f, offset_c, mini_row_index, size;

	// Controla los argumentos al programa
	if ((argc != 3)
		|| ((N = atoi(argv[1])) <= 0)
		|| ((bs = atoi(argv[3])) <= 0)
		|| ((N % bs) != 0))
	{
		printf("\nError en los parámetros. Usage: ./%s N T BS (N debe ser multiplo de BS)\n", argv[0]);
		exit(1);
	}


	size = N*N;

	// Inicializa el randomizador
	time_t t;
	srand((unsigned) time(&t));

	/*****************************************************************/

	// Declara variables para el algoritmo paralelo
	int numProcs, rank, stripSize, blockSize, cellAmount, pos;
	double localAverage1, localAverage2;
	double *blockR1, *blockR2, *blockM, *blockT, *blockRA, *blockRB, *blockC;
	double average[2], localAverage[2];

	// Inicializa MPI
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, NULL); // TODO: guarda con el null, fijate la teoría como lo hace

	// Setea cantidad de hilos y rank
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_size(MPI_COMM_WORLD, &rank);


	if (rank == COORDINATOR) {
		double *C_secuencial, average2, timetick_start_secuencial, timetick_end_secuencial;

		// Aloca memoria para las matrices
		A  = (double*)malloc(sizeof(double)*size); // ordenada por columnas
		B  = (double*)malloc(sizeof(double)*size); // ordenada por columnas
		C  = (double*)malloc(sizeof(double)*size); // ordenada por filas
		C_secuencial = (double*)malloc(sizeof(double)*size); // ordenada por filas
		T  = (double*)malloc(sizeof(double)*size); // ordenada por filas
		M  = (double*)malloc(sizeof(double)*size); // ordenada por filas
		R1 = (double*)malloc(sizeof(double)*size); // ordenada por filas
		R2 = (double*)malloc(sizeof(double)*size); // ordenada por filas
		RA = (double*)malloc(sizeof(double)*size); // ordenada por filas
		RB = (double*)malloc(sizeof(double)*size); // ordenada por filas


		// Inicializa las matrices A, B, T, M, R1, R2, RA, y RB
		for(i = 0; i < size ; i++) {
			A[i] = randFP(0, 10);
			B[i] = randFP(0, 10);
			T[i] = randFP(0, 10);
			M[i] = randFP(0, DOUBLE_PI);
			R1[i] = 0;
			R2[i] = 0;
			RA[i] = 0;
			RB[i] = 0;
		}


		/*********************************** Secuencial ******************************/

		average1 = 0;
		average2 = 0;

		// Inicia el timer
		timetick_start_secuencial = dwalltime();

		// Calcula Rs y su promedio
		for (i = 0; i < size; i++) {
			num = (1 - T[i]);
			aSin = sin(M[i]);
			aCos = cos(M[i]);
			R1[i] = num * (1 - aCos) + (T[i] * aSin);
			R2[i] = num * (1 - aSin) + (T[i] * aCos);

			average1 += R1[i];
			average2 += R2[i];
		}

		// RA = R1 * A
		for (i = 0; i < N; i += bs)
		{
			offset_i = i * N;
			for (j = 0; j < N; j += bs)
			{
				offset_j = j * N;
				cblk = &RA[offset_i + j];

				for  (k = 0; k < N; k += bs)
				{
					ablk = &R1[offset_i + k];
					bblk = &A[offset_j + k];

					for (f = 0; f < bs; f++)
					{
						offset_f = f * N;
						for (c = 0; c < bs; c++)
						{
							offset_c = c * N;
							mini_row_index = offset_f + c;

							for  (h = 0; h < bs; h++)
							{
								cblk[mini_row_index] += ablk[offset_f + h] * bblk[offset_c + h];
							}
						}
					}
				}
			}
		}

		// RB = R2 * B
		for (i = 0; i < N; i += bs)
		{
			offset_i = i * N;
			for (j = 0; j < N; j += bs)
			{
				offset_j = j * N;
				cblk = &RB[offset_i + j];

				for  (k = 0; k < N; k += bs)
				{
					ablk = &R2[offset_i + k];
					bblk = &B[offset_j + k];

					for (f = 0; f < bs; f++)
					{
						offset_f = f * N;
						for (c = 0; c < bs; c++)
						{
							offset_c = c * N;
							mini_row_index = offset_f + c;

							for  (h = 0; h < bs; h++)
							{
								cblk[mini_row_index] += ablk[offset_f + h] * bblk[offset_c + h];
							}
						}
					}
				}
			}
		}


		average1 = (average1 / size) * (average2 / size);

		// Calcula C_secuencial
		for (i = 0; i < size; i++) {
			C_secuencial[i] = T[i] + average1 * (RA[i] + RB[i]);
		}

		// Detiene el timer
		timetick_end_secuencial = dwalltime();

		printf("Tiempo en segundos del secuencial %f\n", timetick_end_secuencial - timetick_start_secuencial);

		// Resetea las matrices R1, R2, RA, RB, y C
		for(i = 0; i < size ; i++) {
			R1[i] = 0;
			R2[i] = 0;
			RA[i] = 0;
			RB[i] = 0;
		}

		average1 = 0;
	}


	/*********************************** MPI ******************************/

	// Setear los bloques
	blockSize = N / numProcs;
	cellAmount = blockSize * N;

	blockR1 = (double*)malloc(sizeof(double)*cellAmount);
	blockR2 = (double*)malloc(sizeof(double)*cellAmount);
	blockM  = (double*)malloc(sizeof(double)*cellAmount);
	blockT  = (double*)malloc(sizeof(double)*cellAmount);
	blockRA = (double*)malloc(sizeof(double)*cellAmount);
	blockRB = (double*)malloc(sizeof(double)*cellAmount);
	blockC  = (double*)malloc(sizeof(double)*cellAmount);

	MPI_Scatter(M, cellAmount, MPI_INT, blockM, cellAmount, MPI_INT, COORDINATOR, MPI_COMM_WORLD);
	MPI_Scatter(T, cellAmount, MPI_INT, blockT, cellAmount, MPI_INT, COORDINATOR, MPI_COMM_WORLD);

	// Inicia el timer
	timetick_start = dwalltime();

	// Broadcastea las matrices que se usan enteras
	MPI_Bcast(A, size, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);
	MPI_Bcast(B, size, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);

#pragma omp parallel
{

	// Calcula Rs
#pragma omp reduction(+:localAverage[0], +:localAverage[1]) private(i, num, aSin, aCos) nowait
	for (i = 0; i < cellAmount; i++) {
		num = (1 - blockT[i]);
		aSin = sin(blockM[i]);
		aCos = cos(blockM[i]);
		blockR1[i] = num * (1 - aCos) + (blockT[i] * aSin);
		blockR2[i] = num * (1 - aSin) + (blockT[i] * aCos);
		localAverage[0] += blockR1[i];
		localAverage[1] += blockR2[i];
	}

	// RA = R1 * A
#pragma omp for private(i, offset_i, j, offset_j, k, ablk, bblk, cblk, f, offset_f, c, offset_c, h, mini_row_index) nowait
	for (i = 0; i < blockSize; i += bs)
	{
		offset_i = i * N;
		for (j = 0; j < N; j += bs)
		{
			offset_j = j * N;
			cblk = &blockRA[offset_i + j];

			for  (k = 0; k < N; k += bs)
			{
				ablk = &blockR1[offset_i + k];
				bblk = &A[offset_j + k];

				for (f = 0; f < bs; f++)
				{
					offset_f = f * N;
					for (c = 0; c < bs; c++)
					{
						offset_c = c * N;
						mini_row_index = offset_f + c;

						for  (h = 0; h < bs; h++)
						{
							cblk[mini_row_index] += ablk[offset_f + h] * bblk[offset_c + h];
						} } } } } }

	// RB = R2 * B
#pragma omp for private(i, offset_i, j, offset_j, k, ablk, bblk, cblk, f, offset_f, c, offset_c, h, mini_row_index) nowait
	for (i = 0; i < blockSize; i += bs)
	{
		offset_i = i * N;
		for (j = 0; j < N; j += bs)
		{
			offset_j = j * N;
			cblk = &blockRB[offset_i + j];

			for  (k = 0; k < N; k += bs)
			{
				ablk = &blockR2[offset_i + k];
				bblk = &B[offset_j + k];

				for (f = 0; f < bs; f++)
				{
					offset_f = f * N;
					for (c = 0; c < bs; c++)
					{
						offset_c = c * N;
						mini_row_index = offset_f + c;

						for  (h = 0; h < bs; h++)
						{
							cblk[mini_row_index] += ablk[offset_f + h] * bblk[offset_c + h];
						} } } } } }


	// Calcula promedios de Rs
#pragma omp single
{
	MPI_Allreduce(localAverage, average, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	average1 = (average[0] / size) * (average[1] / size);
}

	// Calcula C
#pragma omp for private(i)
	for (i = 0; i < cellAmount; i++) {
		blockC[i] = blockT[i] + average1 * (blockRA[i] + blockRB[i]);
	}
}
	MPI_Gather(blockC, cellAmount, MPI_DOUBLE, C, cellAmount, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);

	// Termina MPI
	MPI_Finalize();

	// Detiene el timer
	timetick_end = dwalltime();


	// Free matrices paralelas
	free(blockR1);
	free(blockR2);
	free(blockM);
	free(blockT);
	free(blockRA);
	free(blockRB);
	free(blockC);


	/*****************************************************************/


	if (rank == COORDINATOR) {
		printf("Tiempo en segundos con OpenMP %f\n", timetick_end - timetick_start);

		// Comprueba que C y C_secuencial sean iguales
		int check = 1;
		for (i = 0; i < size; i++) {
			if (fabs(C[i] - C_secuencial[i]) > 0.000001) {
				printf("C: paralelo: %f, secuencial: %f, indice: %d", C[i], C_secuencial[i], i);
				check = 0;
				break;
			}
		}
		if (check) {
			printf("Multiplicacion de matrices resultado correcto\n");
		}
		else {
			printf("Multiplicacion de matrices resultado erroneo\n");
		}

		// Libera memoria
		free(A);
		free(B);
		free(C);
		free(C_secuencial);
		free(T);
		free(M);
		free(R1);
		free(R2);
		free(RA);
		free(RB);
	}

	return(0);
}
