/*
 * Sophie Tan
 * 1402133
 * G100 Mathematics
 *
 * Changed usage of longs to int, as LAPACK operates on ints and could potentially cause overflowing.
 * When outputting extremely small values, they are rounded to 0.
 */

#include <stdlib.h>
#include <stdio.h>
#include <lapacke.h>
#include <math.h>

#define ANSI_RESET  "\e[0m"

#define ANSI_BLACK  "\e[30m"
#define ANSI_RED  "\e[31m"
#define ANSI_GREEN  "\e[32m"
#define ANSI_YELLOW  "\e[33m"
#define ANSI_BLUE  "\e[34m"
#define ANSI_MAGENTA  "\e[35m"
#define ANSI_CYAN  "\e[36m"
#define ANSI_L_GREY  "\e[37m"
#define ANSI_D_GREY "\e[90m"
#define ANSI_L_RED "\e[91m"
#define ANSI_L_GREEN "\e[92m"
#define ANSI_L_YELLOW "\e[93m"
#define ANSI_L_BLUE "\e[94m"
#define ANSI_L_MAGENTA "\e[95m"
#define ANSI_L_CYAN  "\e[96m"
#define ANSI_L_WHITE  "\e[97m"

#define ANSI_BOLD  "\x1B[1m"

#define FLOAT_CMP_RTOL 1e-05
#define FLOAT_CMP_ATOL 1e-08
#define TIME_GRANULARITY 5
#define DEBUG 1

#define CD(i, j) (i + P->I * (j))

#undef I

/* Parameters */
struct params {
    double t_f, t_d, x_R, y_H, gamma_B, T_C, T_w;
    /*
     * I: x steps
     * J: y steps
     * K: t steps
     * S: matrix size (I * J)
     */
    int I, J, K, S;
};
typedef struct params params;

/* Band matrix structure */
struct band_mat {
    int ncol;           /* Number of columns in band matrix             */
    int nbrows;         /* Number of rows (bands in original matrix)    */
    int nbands_up;      /* Number of bands above diagonal               */
    int nbands_low;     /* Number of bands below diagonal               */
    double *array;      /* Storage for the matrix in banded format      */
    /* Internal temporary storage for solving inverse problem           */
    int nbrows_inv;     /* Number of rows of inverse matrix             */
    double *array_inv;  /* Store the inverse if this is generated       */
    int *ipiv;          /* Additional inverse information               */
};
typedef struct band_mat band_mat;

/* Initialise a band matrix of a certain size, allocate memory, and set them parameters. */
int init_band_mat(band_mat *bmat, int nbands_lower, int nbands_upper, int n_columns);

/* Finalise function: should free memory as required */
void finalise_band_mat(band_mat *bmat);

/* Get a pointer to a location in the band matrix, using the row and column indexes of the full matrix. */
double *getp(band_mat *bmat, int row, int column);

/* Set value to a location in the band matrix, using the row and column indexes of the full matrix. */
void setv(band_mat *bmat, int row, int column, double val);

/* Solve the equation Ax = b for a matrix a stored in band format and x and b real arrays */
int solve_Ax_eq_b(band_mat *bmat, double *x, double *b);

/* Print band matrix */
void print_bmat(band_mat *bmat);

/* Print matrix */
void print_mat(double* bmat, params* P);

/* Efficient swap memory function */
void swap_mem(double** a, double** b);

/* Equality check using absolute and relative tolerance for floating-point numbers */
int is_close(double a, double b);

int main(int argc, const char* argv[]) {
    /* Parameters */
    params _;
    params *P = &_;

    /* Read inputs */
    FILE *input = fopen("input.txt", "r");
    if (input == NULL) {
        fprintf(stderr, "Can't open input file\n");
        return 1;
    }
    fscanf(input, "%lg\n%lg\n%d\n%d\n%lg\n%lg\n%lg\n%lg\n%lg\n",
           &P->t_f, &P->t_d, &P->I, &P->J, &P->x_R, &P->y_H, &P->gamma_B, &P->T_C, &P->T_w);
    fclose(input);

    /* Set remaining parameters */
    // TODO: stability
    // TODO: floor vs round
    P->K = (int) floor(P->t_f / P->t_d) * TIME_GRANULARITY;
    P->S = P->I * P->J;

    /* Reading coefficients for T(x, y, 0) and E(x, y, 0) */
    // TODO: any reason not to update E in-place
    FILE *coefficients = fopen("coefficients.txt", "r");
    if (coefficients == NULL) {
        fprintf(stderr, "Can't open coefficients file\n");
        return 1;
    }
    double *T = malloc(P->S * sizeof(double)),
           *T_ = malloc(P->S * sizeof(double)),
           *E = malloc(P->S * sizeof(double)),
           *E_ = malloc(P->S * sizeof(double)),
           *d2Tds2 = malloc(P->S * sizeof(double));
    for (int n = 0; n < P->S; ++n) {
        fscanf(coefficients, "%lg %lg", &T[n], &E[n]);
    }
    fclose(coefficients);

    /* Output file for data */
    FILE *output = fopen("output.txt", "w");
    FILE *error = fopen("errorest.txt", "w");

    /* deltas */
    double dx = P->x_R / (P->I - 1);
    double dy = P->y_H / (P->J - 1);
    double dt = P->t_f / (P->K - 1);

    for (int k = 0; k < P->K; ++k) {
        double t = dt * k;

        if (DEBUG) {
            printf(ANSI_YELLOW "Time: %.2g (%d/%d)\n" ANSI_RESET, t, k, P->K);
            printf(ANSI_GREEN " T:" ANSI_RESET "\n");
            print_mat(T, P);

            printf(ANSI_GREEN " E:" ANSI_RESET "\n");
            print_mat(E, P);
        }

        // TODO: boundary & corners?
        for (int i = 1; i < P->I - 1; ++i) {
            for (int j = 1; j < P->J - 1; ++j) {
                // TODO: Laplacian aok?
                d2Tds2[CD(i, j)] = (T[CD(i + 1, j)] + T[CD(i - 1, j)] - 2 * T[CD(i, j)]) / pow(dx, 2) +
                                   (T[CD(i, j + 1)] + T[CD(i, j - 1)] - 2 * T[CD(i, j)]) / pow(dy, 2);
            }
            d2Tds2[CD(i, 0)] = d2Tds2[CD(i, 1)];
            d2Tds2[CD(i, P->J - 1)] = d2Tds2[CD(i, P->J - 2)];
        }
        for (int j = 0; j < P->J; ++j) {
            d2Tds2[CD(0, j)] = d2Tds2[CD(1, j)];
            d2Tds2[CD(P->I - 1, j)] = d2Tds2[CD(P->I - 1, j)];
        }

        if (DEBUG) {
            printf(ANSI_MAGENTA " d2Tds2:" ANSI_RESET "\n");
            print_mat(d2Tds2, P);
        }

        /* Main updates */
        // TODO: use LAPACK
        for (int s = 0; s < P->S; ++s) {
            double dEdt = -E[s] * P->gamma_B / 2.0 * (1.0 - tanh((T[s] - P->T_C) / P->T_w));
            double dTdt = d2Tds2[s] - dEdt;
            double E_prev = E[s],
                   dE = dEdt * dt,
                   T_prev = T[s],
                   dT = dTdt * dt;
            E_[s] = E_prev + dE;
            T_[s] = T_prev + dT;
        }

        /* Log stuff */
        if (!(k % TIME_GRANULARITY)) {
            for (int i = 0; i < P->I; ++i) {
                for (int j = 0; j < P->J; ++j) {
                    fprintf(output, "%lg %lg %lg %lg %lg\n", t, dx * i, dy * j, T[CD(i, j)], E[CD(i, j)]);
                }
            }
        }

        swap_mem(&T, &T_);
        swap_mem(&E, &E_);
    }

    /* Let there be colour */
    band_mat bmat;
    init_band_mat(&bmat, 2, 2, 10);
    if (DEBUG) {
        printf(ANSI_GREEN " BMAT:" ANSI_RESET "\n");
        print_bmat(&bmat);
    }

    /* Cleanup */
    fclose(output);
    fclose(error);
    finalise_band_mat(&bmat);
    free(T);
    free(E);
    free(T_);
    free(E_);
    free(d2Tds2);

    return 0;
}

/* Initialise a band matrix of a certain size, allocate memory, and set the parameters.  */
int init_band_mat(band_mat *bmat, int nbands_lower, int nbands_upper, int n_columns) {
    bmat->nbrows     = nbands_lower + nbands_upper + 1;
    bmat->ncol       = n_columns;
    bmat->nbands_up  = nbands_upper;
    bmat->nbands_low = nbands_lower;
    bmat->array      = (double *) malloc(sizeof(double) * bmat->nbrows * bmat->ncol);
    bmat->nbrows_inv = bmat->nbands_up * 2 + bmat->nbands_low + 1;
    bmat->array_inv  = (double *) malloc(sizeof(double) * (bmat->nbrows + bmat->nbands_low) * bmat->ncol);
    bmat->ipiv       = (int *) malloc(sizeof(int) * bmat->ncol);
    if (bmat->array == NULL || bmat->array_inv == NULL) {
        return 0;
    }
    /* Initialise array to zero */
    for (int i = 0; i < bmat->nbrows * bmat->ncol; i++) {
        bmat->array[i] = 0.0;
    }
    return 1;
}

/* Finalise function: should free memory as required */
void finalise_band_mat(band_mat *bmat) {
    free(bmat->array);
    free(bmat->array_inv);
    free(bmat->ipiv);
}

/* Get a pointer to a location in the band matrix, using he row and column indexes of the full matrix. */
double *getp(band_mat *bmat, int row, int column) {
    int bandno = bmat->nbands_up + row - column;
    if (row < 0 || column < 0 || row >= bmat->ncol || column >= bmat->ncol) {
        fprintf(stderr, "Indexes out of matrix bounds:\n row %d\n col %d\n", row, column);
        exit(1);
    }
    return &bmat->array[bmat->nbrows * column + bandno];
}

/* Set value to a location in the band matrix, using the row and column indexes of the full matrix. */
void setv(band_mat *bmat, int row, int column, double val) {
    double *valr = getp(bmat, row, column);
    int bandno = bmat->nbands_up + row - column;
    if (bandno < 0 || bandno >= bmat->nbrows) {
        printf(ANSI_YELLOW "Setting (%d, %d) out of band:\n band %d\n" ANSI_RESET,
               row, column, bandno);
    }
    *valr = val;
}

/* Solve the equation Ax = b for a matrix a stored in band format and x and b real arrays */
int solve_Ax_eq_b(band_mat *bmat, double *x, double *b) {
    /* Copy bmat array into the temporary store */
    for(int i = 0; i < bmat->ncol; i++) {
        for (int bandno = 0; bandno < bmat->nbrows; bandno++) {
            bmat->array_inv[bmat->nbrows_inv * i + (bandno+bmat->nbands_low)] = bmat->array[bmat->nbrows * i + bandno];
        }
        x[i] = b[i];
    }

    int nrhs = 1;
    int ldab = bmat->nbands_low * 2 + bmat->nbands_up + 1;
    int info = LAPACKE_dgbsv(LAPACK_COL_MAJOR, bmat->ncol, bmat->nbands_low, bmat->nbands_up, nrhs, bmat->array_inv, ldab, bmat->ipiv, x, bmat->ncol);
    return info;
}

/* Print band matrix */
void print_bmat(band_mat *bmat) {

    /* Column header */
    printf("       ");
    for (int i = 0; i < bmat->ncol; ++i) {
        printf("%11d ", i);
    }
    printf("\n");
    printf("     ┌");
    for (int i = 0; i < bmat->ncol; ++i) {
        printf("────────────");
    }
    printf("─┐\n");

    /* Rows */
    for (int i = 0; i < bmat->ncol; ++i) {
        printf("%4d │ ", i);
        for (int j = 0; j < bmat->ncol; ++j) {
            int bandno = bmat->nbands_up + i - j;
            if (bandno < 0 || bandno >= bmat->nbrows) {
                printf(ANSI_RED "%11.4g " ANSI_RESET, *getp(bmat, i, j));
            } else if (i == j) {
                printf(ANSI_BOLD "%11.4g " ANSI_RESET, *getp(bmat, i, j));
            } else {
                printf("%11.4g ", *getp(bmat, i, j));
            }
        }
        printf("│\n");
    }
    printf("     └");
    for (int i = 0; i < bmat->ncol; ++i) {
        printf("────────────");
    }
    printf("─┘\n");
}

/* Print matrix */
void print_mat(double* mat, params* P) {

    /* Column header */
    printf("       ");
    for (int i = 0; i < P->I; ++i) {
        printf("%11d ", i);
    }
    printf("\n");
    printf("     ┌");
    for (int i = 0; i < P->I; ++i) {
        printf("────────────");
    }
    printf("─┐\n");

    /* Rows */
    for (int i = 0; i < P->J; ++i) {
        printf("%4d │ ", i);
        for (int j = 0; j < P->J; ++j) {
            printf("%11.4g ", mat[CD(i, j)]);
        }
        printf("│\n");
    }
    printf("     └");
    for (int i = 0; i < P->J; ++i) {
        printf("────────────");
    }
    printf("─┘\n");
}

/* Equality check using absolute and relative tolerance for floating-point numbers */
int is_close(double a, double b) {
    return fabs(a - b) <= (FLOAT_CMP_ATOL + FLOAT_CMP_RTOL * fabs(b));
}

/* Efficient swap memory function */
void swap_mem(double **a, double **b) {
    double *temp = *b;
    *b = *a;
    *a = temp;
}
