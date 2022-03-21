import numpy as np

def matrix_B(delta_t, mean_t_minus_one):
    theta = mean_t_minus_one[2][0]
    return np.array([[delta_t*np.cos(theta), 0], [delta_t*np.sin(theta), 0], [0, delta_t]])

def initial_covariance_matrix(variance=0.0001):
    return variance*np.identity(3)

# change variance to small number instead of 0!!!
def motion_model_noise_covariance_matrix_R(variance=0.0001):
    return variance*np.identity(3)

def sensor_model_noise_covariance_matrix_Q(variance=0.0001):
    return variance*np.identity(3)

def Kalman_filter(mean_t_minus_1, cov_matrix_t_minus_1, u_t, z_t, delta_t):
    assert mean_t_minus_1.shape == (3,1), 'Shape of mean vector must be (3,1)'
    assert u_t.shape == (2,1), 'Shape of control vector must be (2,1)'
    assert z_t.shape == (3,1), 'Shape of measurement vector must be (3,1)'

    matrix_A = np.identity(3)
    matrix_C = np.identity(3)

    mean_bar_t = np.dot(matrix_A, mean_t_minus_1) + np.dot(matrix_B(delta_t=delta_t, mean_t_minus_one=mean_t_minus_1), u_t)
    cov_matrix_bar_t = np.matmul(matrix_A, np.matmul(cov_matrix_t_minus_1, np.transpose(matrix_A))) + motion_model_noise_covariance_matrix_R()

    if z_t is None:
        # if no measurement is received, return values without correction
        return mean_bar_t, cov_matrix_bar_t

    # the method will crash at the next step if ALL variances in matrices cov_matrix_t_minus_1, R, and Q are zero
    # in the case all these quanties are zero, the position of the robot can be precisely computed from the last state and
    # any additional control inputs and therefore no correction needs to be made - i.e. the algorithm can return
    # mean_bar_t, cov_matrix_bar_t and would give the perfect position and cov_matrix_bar_t is a matrix with only zeros

    try: 
        # check if matrix in calculation of k_t is invertible
        inverse = np.linalg.inv(
                    np.matmul(matrix_C, 
                        np.matmul(cov_matrix_bar_t, 
                            np.transpose(matrix_C)
                        )
                    ) + sensor_model_noise_covariance_matrix_Q()
            )
    except np.linalg.LinAlgError:
        print('Matrix in calculation of k_t is non-invertible. This means that no noise is existent. Non-corrected values are returned.')
        return mean_bar_t, cov_matrix_bar_t


    k_t = np.matmul(cov_matrix_bar_t, 
            np.matmul(np.transpose(matrix_C), inverse)
        )
    mean_t = mean_bar_t + np.dot(k_t, z_t-np.dot(matrix_C, mean_bar_t))
    cov_matrix_t = np.matmul(np.identity(3) - np.matmul(k_t, matrix_C), cov_matrix_bar_t)
    return mean_t, cov_matrix_t

# code for testing purposes

# mean_zero = np.array([0,0,np.pi]).reshape((3,1))
# cov_matrix_zero = initial_covariance_matrix(variance=0.0001)
# u_zero = np.array([1,0]).reshape((2,1))
# z_zero = np.array([1,0,0]).reshape((3,1))
# del_t = 1

# mean_one, cov_matrix_one = Kalman_filter(mean_t_minus_1 = mean_zero,
#                                             cov_matrix_t_minus_1=cov_matrix_zero,
#                                             u_t=u_zero,
#                                             z_t=z_zero,
#                                             delta_t=del_t)

# print(f'mean one: {mean_one}\n cov_matrix one: {cov_matrix_one}')

# u_one = np.array([-1,0]).reshape((2,1))
# z_one = np.array([0,0,0]).reshape((3,1))

# mean_two, cov_matrix_two = Kalman_filter(mean_t_minus_1 = mean_one,
#                                             cov_matrix_t_minus_1=cov_matrix_one,
#                                             u_t=u_one,
#                                             z_t=z_one,
#                                             delta_t=del_t)

# print(f'mean two: {mean_two}\n cov_matrix two: {cov_matrix_two}')