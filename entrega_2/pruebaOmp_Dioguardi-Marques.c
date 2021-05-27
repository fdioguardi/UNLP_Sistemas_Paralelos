#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

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

/*****************************************************************/

// Main del programa
int main(int argc, char* argv[]){
	double *A, *B, *C, *T, *R, *M, *R1, *R2, *RA, *RB, num, aSin, aCos, timetick_start, timetick_end, *ablk, *bblk, *cblk, NUM_THREADS;
	int N, i, j, k, bs, offset_i, offset_j, row_index, f, c, h, offset_f, offset_c, mini_row_index, id;
	double average1 = 0;
	double average2 = 0;

	// Controla los argumentos al programa
	if ((argc != 4)
		|| ((N = atoi(argv[1])) <= 0)
		|| ((NUM_THREADS = atoi(argv[2])) <= 0)
		|| ((bs = atoi(argv[3])) <= 0)
		|| ((N % bs) != 0))
	{
		printf("\nError en los parámetros. Usage: ./%s N T BS (N debe ser multiplo de BS)\n", argv[0]);
		exit(1);
	}

	// Aloca memoria para las matrices
	int size = N*N;
	A = (double*)malloc(sizeof(double)*size); // ordenada por columnas
	B = (double*)malloc(sizeof(double)*size); // ordenada por columnas
	C = (double*)malloc(sizeof(double)*size); // ordenada por filas
	T = (double*)malloc(sizeof(double)*size); // ordenada por filas
	R = (double*)malloc(sizeof(double)*size); // ordenada por filas
	M = (double*)malloc(sizeof(double)*size); // ordenada por filas
	R1 = (double*)malloc(sizeof(double)*size); // ordenada por filas
	R2 = (double*)malloc(sizeof(double)*size); // ordenada por filas
	RA = (double*)malloc(sizeof(double)*size); // ordenada por filas
	RB = (double*)malloc(sizeof(double)*size); // ordenada por filas

	// Inicializa el randomizador
	time_t t;
	srand((unsigned) time(&t));

	// Inicializa las matrices A, B, T, M, R1, R2, RA, y RB
	for(i = 0; i < size ; i++) {
		A[i] = 1.6;
		B[i] = 12;
		T[i] = 987;
		M[i] = PI/3;
		R1[i] = 0;
		R2[i] = 0;
		RA[i] = 0;
		RB[i] = 0;
	}

	// Inicia el timer
	timetick_start = dwalltime();

	// Setea numero de hilos
	omp_set_num_threads(NUM_THREADS);

	// Calcula Rs y su promedio
#pragma omp parallel
{

	#pragma omp reduction(+:average1, +:average2) private(i, num, aSin, aCos)
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
	#pragma omp for private(i, offset_i, j, offset_j, k, ablk, bblk, cblk, f, offset_f, c, offset_c, h, mini_row_index) nowait
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
		} } } } } }

	// RB = R2 * B
	#pragma omp for private(i, offset_i, j, offset_j, k, ablk, bblk, cblk, f, offset_f, c, offset_c, h, mini_row_index) nowait
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
	} } } } } }

	}

	average1 = (average1 / size) * (average2 / size);

	// Calcula C
	#pragma omp parallel for private(i)
		for (i = 0; i < size; i++) {
			C[i] = T[i] + average1 * (RA[i] + RB[i]);
		}



	// Detiene el timer
	timetick_end = dwalltime();

	printf("Tiempo en segundos %f\n", timetick_end - timetick_start);

	printf("omp %f\n", C[i]);

	// Verifica el resultado de C
	/* int check = 1; */
	/* double correct_result = 1; */
	/* for (i = 0; i < size; i++) { */
	/* 	if (fabs(C[i] - correct_result) > 0.000001) { */
	/* 		check = 0; */
	/* 		break; */
	/* 	} */
	/* } */
	/* if (check) { */
	/* 	printf("Multiplicacion de matrices resultado correcto\n"); */
	/* } */
	/* else { */
	/* 	printf("Multiplicacion de matrices resultado erroneo\n"); */
	/* } */

	// Libera memoria
	free(A);
	free(B);
	free(C);
	free(T);
	free(R);
	free(M);
	free(R1);
	free(R2);
	free(RA);
	free(RB);

	return(0);
}
