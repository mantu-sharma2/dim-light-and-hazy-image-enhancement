import cv2
import numpy as np
import tqdm
import math
import os

UPLOAD_FOLDER = './userImage'


def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res



gamma = 0.5
alpha = 0.3
epsilon = 0.001
kernal = 3

def compile_output(temp_image, text):
    black = [0,0,0]     #---Color of the border---
    constant = cv2.copyMakeBorder(temp_image, 5,5,5,5,cv2.BORDER_CONSTANT,value=black )
    constant = np.asarray(constant).astype(np.uint8)

    violet = None
    if(len(constant.shape)==3):
        violet= np.zeros((50, constant.shape[1], 3), np.uint8)
        violet[:] = (100, 255, 255) 
    else:
        violet= np.zeros((50, constant.shape[1]), np.uint8)
        violet[:] = 100 
        
    violet = np.asarray(violet).astype(np.uint8)

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(violet,text,(30,30), font, 0.75,(0,0,0), 2, 0)
    
    constant = cv2.vconcat((violet, constant))
    return constant

def process_image(img):
    image = []
    for row in tqdm.tqdm(img, desc="Converting to grayscale : "):
        temp = []
        for col in row:
            temp.append(int((int(col[0])+int(col[1])+int(col[2]))/3))  
        temp = np.array(temp)
        image.append(temp)

    image = np.array(image)
    image = image.astype(np.uint8)
    row = len(image)
    col = len(image[0])

    image_norm = []
    for i in tqdm.tqdm(range(len(image)), desc="Normalizing image       : "):
        temp = []
        for j in range(len(image[0])):
            temp.append(image[i][j]/255)
        temp = np.array(temp)
        image_norm.append(temp)

    image_norm = np.array(image_norm)
    image_norm = image_norm.astype(np.uint8)
    # Gamma Correction
    gamma_corrected = []
    for row in tqdm.tqdm(image, desc="Gamma Correction        : "):
        temp = []
        for col in row:
            temp.append(int(255*(col/255)**gamma))
        temp = np.array(temp)
        gamma_corrected.append(temp)

    gamma_corrected = np.array(gamma_corrected)
    gamma_corrected = gamma_corrected.astype(np.uint8)


    # Histogram Equalization
    histo_equalized = cv2.equalizeHist(image)
    histo_equalized = histo_equalized.astype(np.uint8)


    # Alpha bending
    row = len(image)
    col = len(image[0])
    enhanced_intensity_image = []
    for i in tqdm.tqdm(range(row), desc="Alpha bending           : "):
        temp = []
        for j in range(len(image[0])):
            temp.append(int((1-alpha)*gamma_corrected[i][j] + alpha*histo_equalized[i][j]))
        temp = np.array(temp)
        enhanced_intensity_image.append(temp)

    enhanced_intensity_image = np.array(enhanced_intensity_image)
    enhanced_intensity_image = enhanced_intensity_image.astype(np.uint8)

    # Otsu thresholding
    threshold = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)     
    threshold = int(threshold[0])

    W = []
    for i in tqdm.tqdm(range(row), desc="Otsu thresholding       : "):
        temp = []
        for j in range(len(image[0])):
            if image[i][j]<threshold:
                temp.append(1)
            else:
                temp.append(0)
        temp = np.array(temp)
        W.append(temp)

    W = np.array(W)
    W = W.astype(np.uint8)

    # Box filter
    W_box = cv2.boxFilter(W, -1, (kernal,kernal))
    image_norm_box = cv2.boxFilter(image_norm, -1, (kernal,kernal))
    image_box = cv2.boxFilter(image, -1, (kernal,kernal))


    # Calculation of a
    a = []

    for i in tqdm.tqdm(range(row), desc="Calculating a           : "):
        temp = []
        for j in range(col):
            num = 0;
            den = 0;
            for k in range(max(0,i-kernal//2),min(row,i+kernal//2+1),1):
                for l in range(max(0,j-kernal//2),min(col,j+kernal//2+1),1):
                    num += image_norm[k][l]*W[k][l]
                    den += (image_norm[k][l]-image_norm_box[k][l])**2
            constant = 1/(kernal**2)

            temp.append((constant*num - image_norm_box[i][j]*W_box[i][j])/(constant*den + epsilon))
        temp = np.array(temp)
        a.append(temp)

    a = np.array(a)

    row = len(image)
    col = len(image[0])

    b = []
    for i in tqdm.tqdm(range(row), desc="Calculating b: "):
        temp = []
        for j in range(col):
            temp.append(W_box[i][j] - a[i][j]*image_box[i][j])
        temp = np.array(temp)
        b.append(temp)
    b = np.array(b)

    a_box = cv2.boxFilter(a, -1, (kernal,kernal))
    b_box = cv2.boxFilter(b, -1, (kernal,kernal))

    guided_filter = []
    for i in tqdm.tqdm(range(row), desc="Guided filter: "):
        temp = []
        for j in range(col):
            temp.append(a_box[i][j]*image_norm[i][j] + b_box[i][j])
        temp = np.array(temp)
        guided_filter.append(temp)
    guided_filter = np.array(guided_filter)

    final_enhanced_image = []
    for i in tqdm.tqdm(range(row), desc="Enhancing image: "):
        temp = []
        for j in range(col):
            temp.append(guided_filter[i][j]*enhanced_intensity_image[i][j] + (1-guided_filter[i][j])*image[i][j])
        temp = np.array(temp)
        final_enhanced_image.append(temp)
    final_enhanced_image = np.array(final_enhanced_image)

    # cv2.imshow("final Enhanced Image",final_enhanced_image)

    result = []
    for i in tqdm.tqdm(range(row), desc="Converting back to RGB : "):
        temp = []
        for j in range(col):
            t = 0
            if image[i][j]:
                t = image[i][j]
            else:
                t = 1

            r = int(img[i][j][0]*(final_enhanced_image[i][j]/t))
            g = int(img[i][j][1]*(final_enhanced_image[i][j]/t))
            b = int(img[i][j][2]*(final_enhanced_image[i][j]/t))
            temp.append([r,g,b])
        temp = np.array(temp)
        result.append(temp)
    result = np.array(result)

    print("Image processed successfully...")
    # cv2.waitKey();
    return result

def image_inhance(file_name):
    try:
        src = cv2.imread(os.path.join(UPLOAD_FOLDER,file_name));

        I = src.astype('float64')/255;
    
        dark = DarkChannel(I,15);
        A = AtmLight(I,dark);
        te = TransmissionEstimate(I,A,15);
        t = TransmissionRefine(src,te);
        J = Recover(I,t,A,0.1);

        # cv2.imshow("dark",dark);
        # cv2.imshow("t",t);
        # cv2.imshow('I',src);
        # cv2.imshow('J',J);
        cv2.imwrite("dehazed_output.png",J*255);
        img = cv2.imread("dehazed_output.png");
        final_image=process_image(img)
        filename= "processed_"+file_name
        cv2.imwrite(os.path.join(UPLOAD_FOLDER,filename),final_image)
        return filename
    except:
        return None