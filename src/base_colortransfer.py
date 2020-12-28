import numpy as np
import cv2
import typer
import scipy
import scipy.linalg
import scipy.optimize

app = typer.Typer()

img_target = "Donnees/I3.jpg" 
img_source = "Donnees/I9.jpg"



# loading images
targetbgr = cv2.imread(img_target, cv2.IMREAD_UNCHANGED)
sourcebgr = cv2.imread(img_source, cv2.IMREAD_UNCHANGED)

# converting images to LAB space
target = cv2.cvtColor(targetbgr,cv2.COLOR_BGR2LAB)
source = cv2.cvtColor(sourcebgr,cv2.COLOR_BGR2LAB)


def compute_hist_features (image3channels, nbsamples=32):
	[ys, xs] = np.histogram((image3channels[:, :, 0]).flatten(), bins=256)
	ys_c = np.cumsum(ys) / ((image3channels.shape[0]*(image3channels.shape)[1]))
	percents = np.arange(1., 1 + nbsamples) / (nbsamples+1)
	index = np.searchsorted(ys_c, percents)
	return xs[index]


def covariance( image, lambdar ):

   # Computing mean*meanT matrix
   img_mean = np.mean(image, axis=(0, 1))
   img_mean_T = img_mean[np.newaxis, :]    			#reshaping mean_T as (1, 3)  (transposed LAB vector)
   img_mean = np.einsum('ul->lu', img_mean_T)		#reshaping mean as (3, 1) 
   img_mean_cov = np.dot(img_mean, img_mean_T)  	#computing vector product, output has shape (3, 3)
   
   # Computing x*xT matrix

   img = (image[np.newaxis, :]).astype(np.float32)		#converting to float32 & adding dimension
   img = np.einsum('lmnu->mnul', img)					#reshaping the image as (n, m, 3, 1)
   imgT = np.einsum('mnul->mnlu', img)					#reshaping the imageT as (n, m, 1, 3)  (array of transposed LAB vectors)
   img_cov = np.einsum('mnul,mnlh->mnuh', img, imgT)	#computing vector product, output has shape (n, m, 3, 3)

   # Computing covariance matrix
   img_cov = np.sum(img_cov, axis=(0,1))				#sum over the 2 first axes, result has shape (3, 3)
   img_cov = np.divide(img_cov, np.size(image, axis=0) * np.size(image, axis=1))  #normalising

   img_cov = np.subtract(img_cov, img_mean_cov)			#substracting mean*mean_T

   lambdar = 1.
   img_cov[img_cov < lambdar] = lambdar

#    print('Covariance matrix :')
#    print(img_cov)
#    print(' ')
#    print(' ')
   return img_cov


def transfer_func (LI, param):
	tmp = np.arctan(param[0] / param[1])
	return (tmp + np.arctan((LI - param[0]) / param[1])) / (tmp + np.arctan((1 - param[0]) / param[1]))


@app.command()
def simplegaussian():
	# Color Transferbetween Images
	# Erik Reinhard, Michael Ashikhmin, Bruce Gooch,and Peter Shirley

	# computing standart deviation and mean on the 3 channels separatly
	mean_target, std_target = cv2.meanStdDev(target)
	mean_target =  np.hstack(np.around(mean_target,2)) 
	std_target = np.hstack(np.around(std_target,2))

	mean_source, std_source = cv2.meanStdDev(source)
	mean_source = np.hstack(np.around(mean_source,2))
	std_source = np.hstack(np.around(std_source,2))

	# transfering colors
	result = ((source-mean_source)*(std_target/std_source))+mean_target

	# dealing with problematic pixels
	height, width, channel = source.shape
	for i in range(0,height):
		for j in range(0,width):
			for k in range(0,channel):
				x = result[i, j, k]
				x = round(x)
				# boundary check
				x = 0 if x<0 else x
				x = 255 if x>255 else x
				result[i, j, k] = x

	# convert back to RGB and save result
	converted = cv2.cvtColor(result.astype('uint8'),cv2.COLOR_LAB2BGR)
	cv2.imwrite('Donnees/result_Reinhard.jpg',converted)


@app.command()
def multivariategaussian(chromaonly: bool = False, lambdar: float = 0.):
	# The Linear Monge-Kantorovitch Linear Colour Mapping for Example-Based Colour Transfer
	# F.Pitie, A. Kokaram


	# computing mean value on the 3 channels separatly on each image
	mean_target =  np.mean(target, axis = (0 ,1)) 
	mean_source = np.mean(source, axis = (0 ,1)) 

	#computing the covariance matrices
	cov_source = covariance(source, lambdar)
	cov_target = covariance(target, lambdar)

	u_source, s_source, vh_source = np.linalg.svd(cov_source)

	A = np.dot(u_source, np.dot(np.diag(np.power(s_source, -1./2.)), np.linalg.pinv(np.transpose(vh_source)))) # u * diag(s^n) * v^-T
	B = np.dot(u_source, np.dot(np.diag(np.power(s_source, 1./2.)), np.linalg.pinv(np.transpose(vh_source))))  # u * diag(s^n) * v^-T
	C = cov_target

	D = np.dot(B,np.dot(C,B))
	u_D, s_D, vh_D = np.linalg.svd(D)

	# A = COV_in^-1/2
	# B = COV_in^1/2
	# C = COV_target
	# computing the T matrix, with T = A(D^1/2)A and D = (BCB)  (Corresponds to the linear Monge Kantorovic solution)
	T = np.dot(A,
		np.dot(
			np.dot(u_D , np.dot( np.diag(np.power(s_D, 1./2.)) , np.linalg.pinv(np.transpose(vh_D)))) # u * diag(s^n) * v^-T
			, A)
		)

	# print(' ')
	# print('T matrix:')
	# print(T)

	# transfering colors
	result = np.einsum('lmnu->mnul', (np.subtract(source, mean_source))[np.newaxis, :]) #centering values around 0 and converting shape to (m, n, 3, 1)
	result = np.einsum('ij,mnjk->mnik', T, result)										#computing T * centeredLAB      (3, 3) dot (m, n, 3, 1) -> (m, n, 3, 1)
	result = np.squeeze(result)															#arranging shape to (m, n, 3)
	result = np.add(result, mean_target)												#adding target mean

	if(chromaonly):
		# Automatic Content-Aware Color and Tone Stylization 
		# Joon-Young Lee, Kalyan Sunkavalli, Zhe Lin, Xiaohui Shen, In So Kweon

		#Dertermining best match for m and delta 

		nbsamples = 32
		tau = .4


		# inspired of https://github.com/jinyu121/ACACTS/blob/master/transfer/transfer_style.py

		LI = compute_hist_features (source, nbsamples)
		LS = compute_hist_features (target, nbsamples)

		Ltilde = LI + (LS-LI) * tau/np.minimum(tau, np.linalg.norm(LS - LI, np.inf))

		target_func = lambda param: np.power(np.linalg.norm(transfer_func(LI, param) - Ltilde, 2), 2)
		# ( ˆm,ˆδ) = arg min ‖g(LI)− ̃L‖²
		res = scipy.optimize.minimize(target_func, np.random.random_sample([2]), method='CG', options={'gtol': 1e-4})

		#Matching luminance histogram
		result[:, :, 0] = transfer_func(source[:, :, 0], res.x)  # Computing luminance as described in Lee et al.
		#result[:,:,0] = source[:,:,0].astype('float32')		 # Taking luminance as input luminance
	

	# print(' ')
	# print('Max, mean, min :')
	# print(np.max(result, axis=(0, 1)))
	# print(np.mean(result, axis=(0, 1)))
	# print(np.min(result, axis=(0, 1)))

	# print(' ')
	# print('Mean target:')
	# print(mean_target)

	result =  np.clip(result, 0, 255)
	# convert back to RGB and save result
	converted = cv2.cvtColor(result.astype('uint8'),cv2.COLOR_LAB2BGR)
	cv2.imwrite('Donnees/result_MGD.jpg',converted)

	


if __name__ == "__main__":
	app()