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

// Main del programa
int main(int argc, char* argv[]){

	// Inicializa MPI
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, NULL);

	int COORDINATOR = 0;
	int AMOUNT_COMMS = 6;

	int N, size, bs, rowAmount, cellAmount, numProcs, rank;
	double *A, *B, *C, *T, *M, *R1, *R2, *RA, *RB;
	double localAverage[2], average[2];
	double commTimes[AMOUNT_COMMS], maxCommTimes[AMOUNT_COMMS], minCommTimes[AMOUNT_COMMS], commTime, totalTime;
	double num, aSin, aCos, *ablk, *bblk, *cblk;
	int i, j, k, offset_i, offset_j, row_index, f, c, h, offset_f, offset_c, mini_row_index;

	// Setea cantidad de hilos y rank
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Controla los argumentos al programa
	if ((argc != 3)
		|| ((N = atoi(argv[1])) <= 0)
		|| ((bs = atoi(argv[2])) <= 0)
		|| (N % bs != 0))
	{
		printf("\nError en los parámetros. Usage: ./%s N BS (N debe ser multiplo de BS)\n", argv[0]);
		return EXIT_FAILURE;
	}

	rowAmount = N / numProcs;
	cellAmount = rowAmount * N;
	size = N*N;

	if (rowAmount < bs) {
		bs = rowAmount;
	}

	// Inicializa el randomizador
	time_t t;
	srand((unsigned) time(&t));

	/*****************************************************************/

	// Matrices alocadas por todos los nodos
	A  = (double*)malloc(sizeof(double)*size); // ordenada por columnas
	B  = (double*)malloc(sizeof(double)*size); // ordenada por columnas

	if (rank == COORDINATOR) {

		// Aloca memoria para las matrices en el coordinador
		C  = (double*)malloc(sizeof(double)*size); // ordenada por filas
		T  = (double*)malloc(sizeof(double)*size); // ordenada por filas
		M  = (double*)malloc(sizeof(double)*size); // ordenada por filas
		R1 = (double*)malloc(sizeof(double)*size); // ordenada por filas
		R2 = (double*)malloc(sizeof(double)*size); // ordenada por filas
		RA = (double*)malloc(sizeof(double)*size); // ordenada por filas
		RB = (double*)malloc(sizeof(double)*size); // ordenada por filas


		// Inicializa las matrices A, B, T, M, RA, y RB
		for(i = 0; i < size ; i++) {
			A[i] = randFP(0, 10);
			B[i] = randFP(0, 10);
			T[i] = randFP(0, 10);
			M[i] = randFP(0, DOUBLE_PI);
			RA[i] = 0;
			RB[i] = 0;
		}
	}
	else {
		// Aloca memoria en otros nodos
		C  = (double*)malloc(sizeof(double)*cellAmount);
		T  = (double*)malloc(sizeof(double)*cellAmount);
		M  = (double*)malloc(sizeof(double)*cellAmount);
		R1 = (double*)malloc(sizeof(double)*cellAmount);
		R2 = (double*)malloc(sizeof(double)*cellAmount);
		RA = (double*)malloc(sizeof(double)*cellAmount);
		RB = (double*)malloc(sizeof(double)*cellAmount);
	}


	/*********************************** MPI ******************************/


	commTimes[0] =  dwalltime();

	// Reparte las matrices M y T
	MPI_Scatter(M, cellAmount, MPI_DOUBLE, M, cellAmount, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);
	MPI_Scatter(T, cellAmount, MPI_DOUBLE, T, cellAmount, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);

	// Broadcastea las matrices que se usan enteras
	MPI_Bcast(A, size, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);
	MPI_Bcast(B, size, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);

	commTimes[1] = dwalltime();

	#pragma omp parallel
	{

		// Calcula Rs
		#pragma omp reduction(+:localAverage[0], +:localAverage[1]) private(i, num, aSin, aCos) nowait
		for (i = 0; i < cellAmount; i++) {
			num = (1 - T[i]);
			aSin = sin(M[i]);
			aCos = cos(M[i]);
			R1[i] = num * (1 - aCos) + (T[i] * aSin);
			R2[i] = num * (1 - aSin) + (T[i] * aCos);
			localAverage[0] += R1[i];
			localAverage[1] += R2[i];
		}


		// Calcula promedios de Rs
		#pragma omp single
		{
			commTimes[2] = dwalltime();
			MPI_Allreduce(localAverage, average, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			commTimes[3] = dwalltime();

			average[0] = (average[0] / size) * (average[1] / size);
		}


		// RA = R1 * A
		#pragma omp for private(i, offset_i, j, offset_j, k, ablk, bblk, cblk, f, offset_f, c, offset_c, h, mini_row_index) nowait
		for (i = 0; i < rowAmount; i += bs)
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
		} 	} 	} 	} 	} 	}

		// RB = R2 * B
		#pragma omp for private(i, offset_i, j, offset_j, k, ablk, bblk, cblk, f, offset_f, c, offset_c, h, mini_row_index) nowait
		for (i = 0; i < rowAmount; i += bs)
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
		} 	} 	} 	} 	} 	}

		// Calcula C
		#pragma omp for private(i)
		for (i = 0; i < cellAmount; i++) {
			C[i] = T[i] + average[0] * (RA[i] + RB[i]);
		}
	}


	// Junta el resultado
	commTimes[4] = dwalltime();

	MPI_Gather(C, cellAmount, MPI_DOUBLE, C, cellAmount, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);

	commTimes[5] = dwalltime();

	// Totaliza los tiempos de comunicación
	MPI_Reduce(commTimes, minCommTimes, AMOUNT_COMMS, MPI_DOUBLE, MPI_MIN, COORDINATOR, MPI_COMM_WORLD);
	MPI_Reduce(commTimes, maxCommTimes, AMOUNT_COMMS, MPI_DOUBLE, MPI_MAX, COORDINATOR, MPI_COMM_WORLD);


	/*********************************** Secuencial ******************************/

	if (rank == COORDINATOR) {

		double timetick_start_secuencial, *C_secuencial,  timetick_end_secuencial;

		// Aloca matriz secuencial
		C_secuencial = (double*)malloc(sizeof(double)*size); // ordenada por filas

		// Reinicia variables
		for(i = 0; i < size ; i++) {
			RA[i] = 0;
			RB[i] = 0;
		}

		// Inicia el timer
		timetick_start_secuencial = dwalltime();

		// Calcula Rs y su promedio
		for (i = 0; i < size; i++) {
			num = (1 - T[i]);
			aSin = sin(M[i]);
			aCos = cos(M[i]);
			R1[i] = num * (1 - aCos) + (T[i] * aSin);
			R2[i] = num * (1 - aSin) + (T[i] * aCos);

			average[0] += R1[i];
			average[1] += R2[i];
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

		// Calcula el promedio de Rs
		average[0] = (average[0] / size) * (average[1] / size);

		// Calcula C_secuencial
		for (i = 0; i < size; i++) {
			C_secuencial[i] = T[i] + average[0] * (RA[i] + RB[i]);
		}

		// Detiene el timer
		timetick_end_secuencial = dwalltime();

		printf("\nTiempo en segundos del secuencial %f\n", timetick_end_secuencial - timetick_start_secuencial);




		// Comprueba que C y C_secuencial sean iguales
		int check = 1;
		for (i = 0; i < size; i++) {
			if (fabs(C[i] - C_secuencial[i]) > 0.000001) {
				printf("\nC: paralelo: %f, secuencial: %f, indice: %d\n", C[i], C_secuencial[i], i);
				check = 0;
				break;
			}
		}

		if (check) {
			printf("\nMultiplicacion de matrices resultado correcto\n");

			totalTime = maxCommTimes[AMOUNT_COMMS - 1] - minCommTimes[0];
			commTime = 0;
			for (i = 0; i < AMOUNT_COMMS; i += 2) {
				printf("\ntiempo %d - %d = %f\n", i+1, i, maxCommTimes[i + 1] - minCommTimes[i]);
				commTime += (maxCommTimes[i + 1] - minCommTimes[i]);
			}

			printf("\nMultiplicacion de matrices (N=%d)\tTiempo total=%lf\tTiempo comunicacion=%lf\n", N, totalTime, commTime);
		}
		else {
			printf("\nMultiplicacion de matrices resultado erroneo\n");
		}



		// Libera memoria secuencial
		free(C_secuencial);
	}

	// Free matrices paralelas
	free(A);
	free(B);
	free(C);
	free(M);
	free(T);
	free(R1);
	free(R2);
	free(RA);
	free(RB);

	// Termina MPI
	MPI_Finalize();

	return EXIT_SUCCESS;
}
