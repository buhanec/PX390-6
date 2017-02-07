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
#define TIME_GRANULARITY 1000
#define DEBUG
#define GHOST

#define CD(i, j) ((i) + P->I * (j))
#ifdef GHOST
#define CDG(i, j) ((i) + 1 + (P->I + 2) * ((j) + 1))
#define T_(i, j) T[CDG(i, j)]
#else
#define T_(i, j) T[CD(i, j)]
#endif
#define E_(i, j) E[CD(i, j)]
#define d2Tds2_(i, j) d2Tds2[CD(i, j)]

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

#ifdef GHOST
void print_gmat(double *mat, params *P);
#endif

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
    // TODO: remainder is_close
    P->K = (int) floor(P->t_f / P->t_d) * TIME_GRANULARITY;
    P->S = P->I * P->J;

    /* Reading coefficients for T(x, y, 0) and E(x, y, 0) */
    // TODO: any reason not to update E in-place
    FILE *coefficients = fopen("coefficients.txt", "r");
    if (coefficients == NULL) {
        fprintf(stderr, "Can't open coefficients file\n");
        return 1;
    }
#ifdef GHOST
    double *T = malloc((P->I + 2) * (P->J + 2) * sizeof(double)),
           *E = malloc(P->S * sizeof(double)),
           *d2Tds2 = malloc(P->S * sizeof(double));
    for (int s = 0; s < (P->I + 2) * (P->J + 2); ++s) {
        T[s] = NAN;
    }
    for (int i = 0; i < P->I; ++i) {
        for (int j = 0; j < P->J; ++j) {
            fscanf(coefficients, "%lg %lg", &T_(i, j), &E_(i, j));
        }
        T_(i, -1) = T_(i, 1);
        T_(i, P->J) = T_(i, P->J - 2);
    }
    for (int j = 0; j < P->J; ++j) {
        T_(-1, j) = T_(1, j);
        T_(P->I, j) = T_(P->I - 2, j);
    }
#else
    double *T = malloc(P->S * sizeof(double)),
            *E = malloc(P->S * sizeof(double)),
            *d2Tds2 = malloc(P->S * sizeof(double));
    for (int s = 0; s < P->S; ++s) {
        fscanf(coefficients, "%lg %lg", &T[s], &E[s]);
    }
#endif
    fclose(coefficients);

    /* Output file for data */
    FILE *output = fopen("output.txt", "w");
    FILE *error = fopen("errorest.txt", "w");

    /* deltas */
    double dx = P->x_R / (P->I - 1);
    double dy = P->y_H / (P->J - 1);
    double dt = P->t_f / P->K;

    for (int k = 0; k < P->K + 1; ++k) {
        double t = dt * k;

#ifdef DEBUG
        printf(ANSI_YELLOW "Time: %.2g (%d/%d)\n" ANSI_RESET, t, k, P->K);
        if (!(k % TIME_GRANULARITY)) {
            printf(ANSI_L_CYAN " Logging output\n" ANSI_RESET);
        }
        printf(ANSI_GREEN " T:" ANSI_RESET "\n");
#ifdef GHOST
        print_gmat(T, P);
#else
        print_mat(T, P);
#endif
        printf(ANSI_GREEN " E:" ANSI_RESET "\n");
        print_mat(E, P);
#endif

        /* Log stuff */
        if (!(k % TIME_GRANULARITY)) {
            for (int i = 0; i < P->I; ++i) {
                for (int j = 0; j < P->J; ++j) {
                    fprintf(output, "%lg %lg %lg %lg %lg\n", t, dx * i, dy * j, T[CD(i, j)], E[CD(i, j)]);
                }
            }
        }

        // TODO: boundary & corners?
#ifdef GHOST
        /* Update T ghost points */
        for (int i = 0; i < P->I; ++i) {
            T_(i, -1) = T_(i, 1);
            T_(i, P->J) = T_(i, P->J - 2);
        }
        for (int j = 0; j < P->J; ++j) {
            T_(-1, j) = T_(1, j);
            T_(P->I, j) = T_(P->I - 2, j);
        }
        /* Update d2Tds2 */
        for (int i = 0; i < P->I; ++i) {
            for (int j = 0; j < P->J; ++j) {
                d2Tds2_(i, j) = (T_(i + 1, j) + T_(i - 1, j) - 2 * T_(i, j)) / pow(dx, 2) +
                                (T_(i, j + 1) + T_(i, j - 1) - 2 * T_(i, j)) / pow(dy, 2);
            }
        }
#else
        for (int i = 0; i < P->I; ++i) {
            for (int j = 0; j < P->J; ++j) {
                int j_l = j - 1 + 2 * !j,
                        j_h = j + 1 - 2 * (j == P->J - 1),
                        i_l = i - 1 + 2 * !i,
                        i_r = i + 1 - 2 * (i == P->I - 1);
                d2Tds2_(i, j) = (T_(i_r, j) + T_(i_l, j) - 2 * T_(i, j)) / pow(dx, 2) +
                                (T_(i, j_h) + T_(i, j_l) - 2 * T_(i, j)) / pow(dy, 2);
            }
        }
#endif

#ifdef DEBUG
#ifdef GHOST
        printf(ANSI_MAGENTA " Interim T:" ANSI_RESET "\n");
        print_gmat(T, P);
#endif
        printf(ANSI_MAGENTA " d2Tds2:" ANSI_RESET "\n");
        print_mat(d2Tds2, P);
#endif

        /* Update T and E */
#ifdef GHOST
        for (int i = 0; i < P->I; ++i) {
            for (int j = 0; j < P->J; ++j) {
                double tanch = 1.0 + tanh((T_(i, j) - P->T_C) / P->T_w),
                       dEdt = -E_(i, j) * (P->gamma_B / 2.0) * tanch,
                       dTdt = d2Tds2_(i, j) - dEdt,
                       dE = -E_(i, j) * (P->gamma_B / 2.0) * tanch * dt,
                       //dE_Mo = 1.0 + (P->gamma_B / 2.0) * tanch * dt,
                       dT = dTdt * dt;
                E_(i, j) += dE;
                //E_(i, j) /= dE_Mo;
                T_(i, j) += dT;
            }
        }
        for (int s = 0; s < P->S; ++s) {


        }
#else
        for (int s = 0; s < P->S; ++s) {
            double tanch = 1.0 + tanh((T[s] - P->T_C) / P->T_w),
                    dEdt = -E[s] * (P->gamma_B / 2.0) * tanch,
                    dTdt = d2Tds2[s] - dEdt,
                    dE = -E[s] * (P->gamma_B / 2.0) * tanch * dt,
            //dE_Mo = 1.0 + (P->gamma_B / 2.0) * tanch * dt,
                    dT = dTdt * dt;
            E[s] += dE;
            //E[s] /= dE_Mo;
            T[s] += dT;
        }
#endif
    }

    /* Cleanup */
    fclose(output);
    fclose(error);
    free(T);
    free(E);
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

#ifdef GHOST
/* Print matrix with ghost points */
void print_gmat(double *mat, params *P) {

    /* Column header */
    printf("       ");
    for (int i = -1; i < P->I + 1; ++i) {
        printf("%11d ", i);
    }
    printf("\n");
    printf("     ┌");
    for (int i = -1; i < P->I + 1; ++i) {
        printf("────────────");
    }
    printf("─┐\n");

    /* Rows */
    for (int i = -1; i < P->I + 1; ++i) {
        printf("%4d │ ", i);
        for (int j = -1; j < P->J + 1; ++j) {
            if (i < 0 || j < 0 || i == P->I || j == P->J) {
                printf(ANSI_RED);
            }
            printf("%11.4g ", mat[CDG(i, j)]);
            if (i < 0 || j < 0 || i == P->I || j == P->J) {
                printf(ANSI_RESET);
            }
        }
        printf("│\n");
    }
    printf("     └");
    for (int i = -1; i < P->I + 1; ++i) {
        printf("────────────");
    }
    printf("─┘\n");
}
#endif

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
