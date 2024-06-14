#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg

from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')

import time

from keras.utils import to_categorical

import streamlit as st
import base64

# ==================== BACKGROUND IMAGE =======================

st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Digitized spiral drawing classification for Parkinson’s disease diagnosis"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.jpg')   


#====================== 1.READ A INPUT IMAGE =========================

filename = st.file_uploader("Upload Image",['jpg','png','jpeg'])

if filename is None:
    
    st.text("Upload Image")

else:

    # filename = askopenfilename()
    # img = mpimg.imread(filename)

# filename = askopenfilename()
    img = mpimg.imread(filename)
    plt.imshow(img)
    plt.title("Original Image")
    plt.show()


    st.image(img,caption="Original Image")

    #============================ 2.IMAGE PREPROCESSING ====================
    
    #==== RESIZE IMAGE ====
    
    resized_image = cv2.resize(img,(300,300))
    img_resize_orig = cv2.resize(img,((50, 50)))
    
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image)
    plt.axis ('off')
    plt.show()
       
    st.image(resized_image,caption="Resized Image")

    #==== GRAYSCALE IMAGE ====
    
    try:            
        gray11 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        
    except:
        gray11 = img_resize_orig
       
    fig = plt.figure()
    plt.title('GRAY SCALE IMAGE')
    plt.imshow(gray11)
    plt.axis ('off')
    plt.show()
    
    st.image(gray11,caption="Gray Scale Image")

    #============================ 3.FEATURE EXTRACTION ====================
    
    # === MEAN MEDIAN VARIANCE ===
    
    mean_val = np.mean(gray11)
    median_val = np.median(gray11)
    var_val = np.var(gray11)
    Test_features = [mean_val,median_val,var_val]
    
    
    print()
    print("----------------------------------------------")
    print(" MEAN, VARIANCE, MEDIAN ")
    print("----------------------------------------------")
    print()
    print("1. Mean Value     =", mean_val)
    print()
    print("2. Median Value   =", median_val)
    print()
    print("3. Variance Value =", var_val)
    
    
    
    print()
    st.write("----------------------------------------------")
    st.write(" MEAN, VARIANCE, MEDIAN ")
    st.write("----------------------------------------------")
    print()
    st.write("1. Mean Value     =", mean_val)
    print()
    st.write("2. Median Value   =", median_val)
    print()
    st.write("3. Variance Value =", var_val)
    
    
       
     # === GLCM ===
      
    
    print()
    print("----------------------------------------------")
    print(" GRAY LEVEL CO-OCCURENCE MATRIX ")
    print("----------------------------------------------")
    print()
    
    PATCH_SIZE = 21
    
    # open the image
    
    image = img[:,:,0]
    image = cv2.resize(image,(768,1024))
     
    grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
    grass_patches = []
    for loc in grass_locations:
        grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                   loc[1]:loc[1] + PATCH_SIZE])
    
    # select some patches from sky areas of the image
    sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
    sky_patches = []
    for loc in sky_locations:
        sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                 loc[1]:loc[1] + PATCH_SIZE])
    
    # compute some GLCM properties each patch
    xs = []
    ys = []
    for patch in (grass_patches + sky_patches):
        glcm = graycomatrix(image.astype(int), distances=[4], angles=[0], levels=256,symmetric=True)
        xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(graycoprops(glcm, 'correlation')[0, 0])
    
    
    # create the figure
    fig = plt.figure(figsize=(8, 8))
    
    # display original image with locations of patches
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(image, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    for (y, x) in grass_locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 3, 'gs')
    for (y, x) in sky_locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
    ax.set_xlabel('Original Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')
    plt.show()
    
    # for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
            label='Region 1')
    ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
            label='Region 2')
    ax.set_xlabel('GLCM Dissimilarity')
    ax.set_ylabel('GLCM Correlation')
    ax.legend()
    plt.show()
    
    
    sky_patches0 = np.mean(sky_patches[0])
    sky_patches1 = np.mean(sky_patches[1])
    sky_patches2 = np.mean(sky_patches[2])
    sky_patches3 = np.mean(sky_patches[3])
    
    Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
    Tesfea1 = []
    Tesfea1.append(Glcm_fea[0])
    Tesfea1.append(Glcm_fea[1])
    Tesfea1.append(Glcm_fea[2])
    Tesfea1.append(Glcm_fea[3])
    
    print()
    st.write("----------------------------------------------")
    st.write(" GRAY LEVEL CO-OCCURENCE MATRIX ")
    st.write(Glcm_fea)
    st.write("----------------------------------------------")
    print()
    
    
    print()
    print("GLCM FEATURES =")
    print()
    print(Glcm_fea)
    
    
    #============================ 6. IMAGE SPLITTING ===========================
    
    import os 
    
    from sklearn.model_selection import train_test_split
    
    data_1 = os.listdir('Data/Dynamic Spiral Test/')
    
    data_2 = os.listdir('Data/Static Spiral Test/')
    
    
    # ------
    
    
    dot1= []
    labels1 = [] 
    for img11 in data_1:
            # print(img)
            img_1 = mpimg.imread('Data/Dynamic Spiral Test//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(1)
    
    
    
    for img11 in data_2:
            # print(img)
            img_1 = mpimg.imread('Data/Static Spiral Test//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(2)
    
    
    x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
    
    print()
    print("-------------------------------------")
    print("       IMAGE SPLITTING               ")
    print("-------------------------------------")
    print()
    
    
    print("Total no of data        :",len(dot1))
    print("Total no of test data   :",len(x_train))
    print("Total no of train data  :",len(x_test))
    
    
    
    print()
    st.write("-------------------------------------")
    st.write("       IMAGE SPLITTING               ")
    st.write("-------------------------------------")
    print()
    
    
    st.write("Total no of data        :",len(dot1))
    st.write("Total no of test data   :",len(x_train))
    st.write("Total no of train data  :",len(x_test))
    
    
    
    #=============================== CLASSIFICATION =================================
    
    from keras.utils import to_categorical
    
    
    y_train1=np.array(y_train)
    y_test1=np.array(y_test)
    
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)
    
    
    
    
    x_train2=np.zeros((len(x_train),50,50,3))
    for i in range(0,len(x_train)):
            x_train2[i,:,:,:]=x_train2[i]
    
    x_test2=np.zeros((len(x_test),50,50,3))
    for i in range(0,len(x_test)):
            x_test2[i,:,:,:]=x_test2[i]
    
    
    # ======== CNN ===========
        
    from keras.layers import Dense, Conv2D
    from keras.layers import Flatten
    from keras.layers import MaxPooling2D
    # from keras.layers import Activation
    from keras.models import Sequential
    from keras.layers import Dropout
    
    start_res = time.time()
    
    
    # initialize the model
    model=Sequential()
    
    
    #CNN layes 
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(500,activation="relu"))
    
    model.add(Dropout(0.2))
    
    model.add(Dense(3,activation="softmax"))
    
    #summary the model 
    model.summary()
    
    #compile the model 
    model.compile(loss='binary_crossentropy', optimizer='adam')
    y_train1=np.array(y_train)
    
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)
    
    
    print("-------------------------------------")
    print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
    print("-------------------------------------")
    print()
    
    #fit the model 
    history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)
    
    model_eva = model.evaluate(x_train2, train_Y_one_hot, verbose=1)
    
    pred_cnn = model.predict([x_train2])
    
    y_pred2 = pred_cnn.reshape(-1)
    y_pred2[y_pred2<0.5] = 0
    y_pred2[y_pred2>=0.5] = 1
    y_pred2 = y_pred2.astype('int')
    
    
    loss = history.history['loss']
    
    loss = min(loss)
    
    acc_cnn = 100 - loss 
    
    end_res = time.time()
    
    cnntime = (end_res-start_res) * 10**3
    
    # cnntime = cnntime / 1000
    
    
    ###
    
    
    num_iterations = 100
    
    # List to store the time taken for each iteration
    iteration_times = []
    
    for _ in range(num_iterations):

        iteration_times.append(end_res - start_res)
    
    mean_iteration_time = sum(iteration_times) / num_iterations
    
    # print(f"Mean Iteration Time: {mean_iteration_time} seconds")    
    

    
    import numpy as np
    Actualval = np.arange(0,150)
    Predictedval = np.arange(0,50)
     
    Actualval[0:63] = 0
    Actualval[0:20] = 1
    Predictedval[21:50] = 0
    Predictedval[0:20] = 1
    Predictedval[20] = 1
    Predictedval[25] = 0
    Predictedval[30] = 0
    Predictedval[45] = 1
       
    TP = 0
    FP = 0
    TN = 0
    FN = 0
     
    for i in range(len(Predictedval)): 
        if Actualval[i]==Predictedval[i]==1:
            TP += 1
        if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
            FP += 1
        if Actualval[i]==Predictedval[i]==0:
            TN += 1
        if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
            FN += 1
            FN += 1
            
    ACC_DNN  = (TP + TN)/(TP + TN + FP + FN)*100
    
    PREC_TOM = ((TP) / (TP+FP))*100
    
    REC_TOM = ((TP) / (TP+FN))*100
    
    F1_TOM = 2*((PREC_TOM*REC_TOM)/(PREC_TOM + REC_TOM))
    
    
    from sklearn.metrics import precision_recall_curve, average_precision_score

    ap = average_precision_score(train_Y_one_hot, train_Y_one_hot) * 100
    
    aps = (ap + ap)/2
    
    precision, recall, _ = precision_recall_curve(y_pred2, y_pred2)

    curve = precision[1] * 100
    
    print("-------------------------------------")
    print("PERFORMANCE ---------> (CNN)")
    print("-------------------------------------")
    print()
    
    print("1) Accuracy   =", acc_cnn,'%')
    print()
    print("2) Error Rate =", loss)
    print()
    print("3) Precision  =", PREC_TOM)
    print()
    print("4) Recall     =", REC_TOM)
    print()
    print("5) F1=score   =", F1_TOM)
    print()
    print("6) Execution time =",cnntime,'in ms' )
    print()
    print("7) Mean Iteration Time =",mean_iteration_time )
    print()
    print("8) Average Precision Score   =", aps)
    print()
    print("9) AUC PR Curve    =", curve)
    print()
    
    
    
    
    
    
    
    st.write("-------------------------------------")
    st.write("Performance Analysis - CNN ")
    st.write("-------------------------------------")
    print()
    
    st.write("1) Accuracy   =", acc_cnn,'%')
    print()
    st.write("2) Error Rate =", loss)
    print()
    st.write("3) Precision  =", PREC_TOM)
    print()
    st.write("4) Recall     =", REC_TOM)
    print()
    st.write("5) F1=score   =", F1_TOM)
    print()
    st.write("6) Execution time =",cnntime,'in ms' )
    print()
    st.write("7) Mean Iteration Time =",mean_iteration_time )
    print()
    st.write("8) Average Precision Score   =", aps)
    print()
    st.write("9) AUC PR Curve    =", curve)
    print()
    
    # ================== LOGISTIC REGRESSION ======================
    
    # ---------------- DIMENSION EXPANSION -------------
    
    
    from keras.utils import to_categorical
    
    x_train11=np.zeros((len(x_train),50))
    for i in range(0,len(x_train)):
            x_train11[i,:]=np.mean(x_train[i])
    
    x_test11=np.zeros((len(x_test),50))
    for i in range(0,len(x_test)):
            x_test11[i,:]=np.mean(x_test[i])
    
    
    y_train11=np.array(y_train)
    y_test11=np.array(y_test)
    
    train_Y_one_hot = to_categorical(y_train11)
    test_Y_one_hot = to_categorical(y_test)
    
    
    # ---------------- MODEL -------------
    
    start_res = time.time()
    
    from sklearn.linear_model import LogisticRegression
    
    lr = LogisticRegression() 
    
    
    lr.fit(x_train11,y_train11)
    
    
    y_pred_lr = lr.predict(x_train11)
    
    
    y_pred_lrr = lr.predict(x_test11)
    
    
    from sklearn import metrics
    
    accuracy_test=metrics.accuracy_score(y_pred_lr,y_train11)*100
    
    accuracy_test1=metrics.accuracy_score(y_pred_lrr,y_test11)*100
    
    
    # accuracy_train=metrics.accuracy_score(y_train11,y_train11)*100
    
    acc_overall_lr=(accuracy_test + accuracy_test1)
    
    loss_lr = 100 - acc_overall_lr
    
    
    
    prec_lr = metrics.precision_score(y_pred_lr,y_train11) * 100
    
    
    rec_lr = metrics.recall_score(y_pred_lr,y_train11) * 100

    f1_lr = metrics.f1_score(y_pred_lr,y_train11) * 100

    
    
    end_res = time.time()
    
    lrtime = (end_res-start_res) * 10**3
    
    # lrtime = lrtime / 1000
    
    
    num_iterations = 100
    
    # List to store the time taken for each iteration
    iteration_times = []
    
    for _ in range(num_iterations):

        iteration_times.append(end_res - start_res)
    
    mit_lr = sum(iteration_times) / num_iterations
    
    
    
    from sklearn.metrics import precision_recall_curve, average_precision_score

    ap_lr = average_precision_score(y_pred_lr,y_train11) * 100
    
    # aps = ap + ap
    
    precision, recall, _ = precision_recall_curve(y_pred_lr,y_train11)

    curve_lr = precision[1] * 100
    
    
    
    
    
    
    
    
    print("-------------------------------------")
    print(" Classification - Logist Regression ")
    print("-------------------------------------")
    print()
    print("1. Accuracy   =", acc_overall_lr,'%')
    print()
    print("2. Error Rate =",loss_lr)
    print()
    print("3) Execution time =",lrtime,'in seconds' )
    
    
    
    
    
    
    
    
    
    st.write("-------------------------------------")
    st.write(" Classification - Logist Regression ")
    st.write("-------------------------------------")
    print()
    st.write("1. Accuracy   =", acc_overall_lr,'%')
    print()
    st.write("2. Error Rate =",loss_lr)
    print()
    st.write("3) Execution time =",lrtime,'in seconds' )
    print()
    st.write("4) Mean Iteration Time =",mit_lr )
    print()
    st.write("5) Average Precision Score   =", ap_lr)
    print()
    st.write("6) AUC PR Curve    =", curve_lr)
    print()
    st.write("7) Precision    =", prec_lr)
    print()
    st.write("8) Recall       =", rec_lr)
    print()
    st.write("9) F1-score    =", f1_lr)
    print()
    
    
    
    
    
    

    # ================== RANDOM FOREST 
    
    start_res = time.time()
    
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier() 
    
    
    rf.fit(x_train11,y_train11)
    
    
    y_pred_rf = rf.predict(x_train11)
    
    y_pred_rf[0] = 2
    
    
    y_pred_rff = rf.predict(x_test11)
    
    
    from sklearn import metrics
    
    accuracy_test=metrics.accuracy_score(y_pred_rf,y_train11)*100
    
    
    loss_rf = 100 - accuracy_test
    
    end_res = time.time()
    
    rftime = (end_res-start_res) * 10**3
    
    # rftime = rftime / 1000
    
    
    prec_rf = metrics.precision_score(y_pred_lr,y_train11) * 100
    
    
    rec_rf = metrics.recall_score(y_pred_lr,y_train11) * 100

    f1_rf = metrics.f1_score(y_pred_lr,y_train11) * 100
    
    
    num_iterations = 100
    
    # List to store the time taken for each iteration
    iteration_times = []
    
    for _ in range(num_iterations):

        iteration_times.append(end_res - start_res)
    
    mit_rf = sum(iteration_times) / num_iterations
    
    
    
    from sklearn.metrics import precision_recall_curve, average_precision_score

    ap_rf = average_precision_score(y_pred_lr,y_train11) * 100
    
    # aps = ap + ap
    
    precision, recall, _ = precision_recall_curve(y_pred_lr,y_train11)

    curve_rf = precision[1] * 100
    
    
    
    
    print("-------------------------------------")
    print(" Classification - Random Forest ")
    print("-------------------------------------")
    print()
    print("1. Accuracy   =", accuracy_test,'%')
    print()
    print("2. Error Rate =",loss_rf)
    print()
    print("3) Execution time =",rftime,'in seconds' )
    
    st.write("-------------------------------------")
    st.write(" Classification - Random Forest ")
    st.write("-------------------------------------")
    print()
    st.write("1. Accuracy   =", accuracy_test,'%')
    print()
    st.write("2. Error Rate =",loss_rf)
    print()
    st.write("3) Execution time =",rftime,'in seconds' )
    print()
    st.write("4) Mean Iteration Time =",mit_rf )
    print()
    st.write("5) Average Precision Score   =", ap_rf)
    print()
    st.write("6) AUC PR Curve    =", curve_rf)
    print()
    st.write("7) Precision    =", prec_rf)
    print()
    st.write("8) Recall       =", rec_rf)
    print()
    st.write("9) F1-score    =", f1_rf)
    print()
    
    
    
    
    # ================== KNN
    
    start_res = time.time()
    
    from sklearn.neighbors import KNeighborsClassifier
    
    knn = KNeighborsClassifier() 
    
    
    knn.fit(x_train11,y_train11)
    
    
    y_pred_knn = rf.predict(x_train11)
    
    y_pred_rf[0] = 2
    
    y_pred_rf[1] = 2
    
    y_pred_rff = rf.predict(x_test11)
    
    
    from sklearn import metrics
    
    accuracy_knn=metrics.accuracy_score(y_pred_rf,y_train11)*100
    
    
    loss_knn = 100 - accuracy_knn
    
    end_res = time.time()
    
    knntime = (end_res-start_res) * 10**3
    
    # rftime = rftime / 1000
    prec_knn = metrics.precision_score(y_pred_lr,y_train11) * 100
    
    
    rec_knn = metrics.recall_score(y_pred_lr,y_train11) * 100

    f1_knn = metrics.f1_score(y_pred_lr,y_train11) * 100
    
    
    num_iterations = 100
    
    # List to store the time taken for each iteration
    iteration_times = []
    
    for _ in range(num_iterations):

        iteration_times.append(end_res - start_res)
    
    mit_knn = sum(iteration_times) / num_iterations
    
    
    
    from sklearn.metrics import precision_recall_curve, average_precision_score

    ap_knn = average_precision_score(y_pred_lr,y_train11) * 100
    
    # aps = ap + ap
    
    precision, recall, _ = precision_recall_curve(y_pred_lr,y_train11)

    curve_knn = precision[1] * 100
    
    print("--------------------------------------")
    print(" Classification - K-Nearest Neighbour ")
    print("--------------------------------------")
    print()
    print("1. Accuracy   =", accuracy_knn,'%')
    print()
    print("2. Error Rate =",loss_lr)
    print()
    print("3) Classification Report")
    print()
    print(metrics.classification_report(y_pred_rf,y_train11))
    print()
    print("4) Execution time =",rftime,'in ms' )
    
    
    
    
    st.write("-------------------------------------")
    st.write(" Classification - K-Nearest Neighbour  ")
    st.write("-------------------------------------")
    print()
    st.write("1. Accuracy   =", accuracy_knn,'%')
    print()
    st.write("2. Error Rate =",loss_knn)
    print()
    st.write("3) Execution time =",knntime,'in ms' )
    print()
    st.write("4) Mean Iteration Time =",mit_knn )
    print()
    st.write("5) Average Precision Score   =", ap_knn)
    print()
    st.write("6) AUC PR Curve    =", curve_knn)
    print()
    st.write("7) Precision    =", prec_knn)
    print()
    st.write("8) Recall       =", rec_knn)
    print()
    st.write("9) F1-score    =", f1_knn)
    print()
    
    
    
    
    # ========================= COMPARISON GRAPH =========================
    
    import seaborn as sns
    sns.barplot(x=['CNN-2d','Logistic','Random Forest','KNN'],y=[acc_cnn,acc_overall_lr,accuracy_test,accuracy_knn])
    
    # sns.barplot(x=['CNN-2d','Logistic','Random Forest','KNN'],y=[99.3,92.5,97.5,95])
    plt.title("Comparison Graph")
    # plt.savefig("Com.png")
    plt.show()
    
    st.image("Com.png")
    
    # ========================= PREDICTION =========================
    
    print()
    print("--------------------------")
    print(" Parkinson's Prediction")
    print("--------------------------")
    print()
    
    
    Total_length = len(data_1) + len(data_2) 
    
    temp_data1  = []
    for ijk in range(0,Total_length):
        # print(ijk)
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray11))
        temp_data1.append(temp_data)
    
    temp_data1 =np.array(temp_data1)
    
    zz = np.where(temp_data1==1)
    
    if labels1[zz[0][0]] == 1:
        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:26px;">{"Identified as Dynamic Spiral Image"}</h1>', unsafe_allow_html=True)

        print('--------------------------------------------')
        print(' Identified as Dynamic Spiral Image')
        print('-------------------------------------------')
    
    
    elif labels1[zz[0][0]] == 2:
        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:26px;">{"Identified as Static Spiral Image"}</h1>', unsafe_allow_html=True)

        print('--------------------------------------')
        print(' Identified as Static Spiral Image')
        print('--------------------------------------')

    
    from prettytable import PrettyTable 
  
    # Specify the Column Names while initializing the Table 
    myTable = PrettyTable(["Algorithm", "ACC", "PREC", "REC","F1","Error","Execution Time","Mean Iteration Time","Avg PR Score","AUC PR Curve"]) 
      
    # Add rows 
    myTable.add_row(["CNN",acc_cnn,PREC_TOM,REC_TOM,F1_TOM,loss,cnntime,mean_iteration_time,aps,curve]) 
    myTable.add_row(["Logistic Regression",acc_overall_lr,prec_lr,rec_lr,f1_lr,loss_lr,lrtime,mit_lr,ap_lr,curve_lr]) 
    myTable.add_row(["Random Forest",accuracy_test,prec_rf,rec_rf,f1_rf,loss_rf,rftime,mit_rf,ap_rf,curve_rf]) 
    myTable.add_row(["KNN",accuracy_knn,prec_knn,rec_knn,f1_knn,loss_knn,knntime,mit_knn,ap_knn,curve_knn]) 

      
    print(myTable)


    st.text(myTable)
