import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from tkinter import *
from tkinter import messagebox

Data=pd.read_csv('diabetes.csv')
X = Data.drop(['Outcome'],axis=1)
y = Data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False,random_state = 42)
#xay dung mo hinh svc
svc=SVC()
svc.fit(X_train,y_train)

#Xay dung mo hinh id3
clf_gini = DecisionTreeClassifier(criterion='entropy',max_depth=8, random_state=42)
clf_gini.fit(X_train,y_train)

#xay dung mo hinh gausss
model_gauss=GaussianNB()
model_gauss.fit(X_train,y_train)

#xay dung mo hinh Per()
model_per=Perceptron()
model_per.fit(X_train,y_train)

#ung dung mo hinh
dd_svc=svc.predict(X_test)
dd_clf=clf_gini.predict(X_test)
dd_gauss=model_gauss.predict(X_test)
dd_per=model_per.predict(X_test)

#hien thi ket qua va danh gia
# print(dd_svc)
# print(metrics.confusion_matrix(y_test,dd_svc))
# print(metrics.classification_report(y_test,dd_svc))

# print(dd_clf)
# print(metrics.confusion_matrix(y_test,dd_clf))
# print(metrics.classification_report(y_test,dd_clf))

# print(dd_gauss)
# print(metrics.confusion_matrix(y_test,dd_gauss))
# print(metrics.classification_report(y_test,dd_gauss))
print(y_test[1]/dd_gauss)

# print(dd_per)
# print(metrics.confusion_matrix(y_test,dd_per))
# print(metrics.classification_report(y_test,dd_per))

####################

root=Tk()
root.title("Dự đoán bệnh tiểu đường")
root.configure(bg='#FFB6C1')
root.geometry("1000x650")

label_title=Label(root,text="Dự đoán bệnh tiểu đường",font = ("Quicksand", 17),foreground="black",bg='#FFB6C1')
label_title.pack(side = "top")

label_title_input=Label(root,text="Nhập các chỉ số",font=("Quicksand",15),bg='#FFB6C1')
label_title_input.place(x = 40 , y = 40)

label1=Label(root,text="Pregnancies ",font=("Quicksand",12),bg='#FFB6C1')
label1.place(x = 40 , y = 80)
a1=Entry(root,width=20,borderwidth=2)
a1.place(x = 270 , y =80)

label2=Label(root,text="Glucose",font=("Quicksand",12),bg='#FFB6C1')
label2.place(x = 40 , y = 120)
b1=Entry(root,width=20,borderwidth=2)
b1.place(x = 270 , y =120)

label3=Label(root,text="BloodPressure ",font=("Quicksand",12),bg='#FFB6C1')
label3.place(x = 40 , y = 160)
c1=Entry(root,width=20,borderwidth=2)
c1.place(x = 270 , y =160)

label4=Label(root,text="SkinThickness",font=("Quicksand",12),bg='#FFB6C1')
label4.place(x = 40 , y = 200)
d1=Entry(root,width=20,borderwidth=2)
d1.place(x = 270 , y =200)

label5=Label(root,text="Insulin",font=("Quicksand",12),bg='#FFB6C1')
label5.place(x = 500 , y = 80)
e1=Entry(root,width=20,borderwidth=2)
e1.place(x = 730 , y =80)

label6=Label(root,text="BMI",font=("Quicksand",12),bg='#FFB6C1')
label6.place(x = 500 , y = 120)
f1=Entry(root,width=20,borderwidth=2)
f1.place(x = 730 , y =120)

label7=Label(root,text="DiabetesPedigreeFunction",font=("Quicksand",12),bg='#FFB6C1')
label7.place(x = 500 , y = 160)
g1=Entry(root,width=20,borderwidth=2)
g1.place(x = 730 , y =160)

label8=Label(root,text="Age",font=("Quicksand",12),bg='#FFB6C1')
label8.place(x = 500 , y = 200)
h1=Entry(root,width=20,borderwidth=2)
h1.place(x = 730 , y =200)
a=''

label_title_du_doan=Label(root,text="Dự đoán",font=("Quicksand",15),bg='#FFB6C1')
label_title_du_doan.place(x = 40 , y = 250)
gta1='';gtb1='';gtc1='';gtd1='';gte1='';gtf1='';gtg1='';gth1=''

def checkInput():
    global gta1,gtb1,gtc1,gtd1,gte1,gtf1,gtg1,gth1
    gta1=a1.get()
    gtb1=b1.get()
    gtc1=c1.get()
    gtd1=d1.get()
    gte1=e1.get()
    gtf1=f1.get()
    gtg1=g1.get()
    gth1=h1.get()
    if(gta1=="" or gtb1=='' or gtc1=='' or gtd1=='' or gte1=='' or gtf1=='' or gtg1=='' or gth1==''):
        messagebox.showerror('Error','Vui lòng điền đầy đủ thông tin')
        return False
    else:
        gta1=float(gta1)
        gtb1=float(gtb1)
        gtc1=float(gtc1)
        gtd1=float(gtd1)
        gte1=float(gte1)
        gtf1=float(gtf1)
        gtg1=float(gtg1)
        gth1=float(gth1) 
        return True
def forecast_ID3():
    if checkInput()==True:
        tb_kq_dd.delete("1.0","end")
        input_id3=np.array([gta1,gtb1,gtc1,gtd1,gte1,gtf1,gtg1,gth1]).reshape(1, -1)
        kq_dd_id3=clf_gini.predict(input_id3)
        if kq_dd_id3==[1]: text_id3='Bị bệnh tiểu đường' 
        else: text_id3=' Không bị bệnh tiểu đường' 
        tb_kq_dd.insert(END,'Dựa theo ID3 dự đoán: ')
        tb_kq_dd.insert(END,text_id3)
def forecast_SVC():
    if checkInput()==True:
        tb_kq_dd.delete("1.0","end")
        input_svc=np.array([gta1,gtb1,gtc1,gtd1,gte1,gtf1,gtg1,gth1]).reshape(1, -1)
        kq_dd_svc=svc.predict(input_svc)
        if kq_dd_svc==[1]: text_svc='Bị bệnh tiểu đường' 
        else: text_svc=' Không bị bệnh tiểu đường' 
        tb_kq_dd.insert(END,'Dựa theo SVC dự đoán: ')
        tb_kq_dd.insert(END,text_svc)
def forecast_Gauss():
    if checkInput()==True:
        tb_kq_dd.delete("1.0","end")
        input_gauss=np.array([gta1,gtb1,gtc1,gtd1,gte1,gtf1,gtg1,gth1]).reshape(1, -1)
        kq_dd_gauss=model_gauss.predict(input_gauss)
        if kq_dd_gauss==[1]: text_gauss='Bị bệnh tiểu đường' 
        else: text_gauss=' Không bị bệnh tiểu đường' 
        tb_kq_dd.insert(END,'Dựa theo GaussianNB dự đoán: ')
        tb_kq_dd.insert(END,text_gauss)
def forecast_Per():
    if checkInput()==True:
        tb_kq_dd.delete("1.0","end")
        input_per=np.array([gta1,gtb1,gtc1,gtd1,gte1,gtf1,gtg1,gth1]).reshape(1, -1)
        kq_dd_per=model_per.predict(input_per)
        if kq_dd_per==[1]: text_per='Bị bệnh tiểu đường' 
        else: text_per=' Không bị bệnh tiểu đường' 
        tb_kq_dd.insert(END,"Dựa theo Perceptron dự đoán: ")
        tb_kq_dd.insert(END,text_per)

bt_dd_per=Button(root,text='Dự đoán Perceptron',foreground="blue",width=20,command=forecast_Per).place(x=70,y=300)
bt_dd_svc=Button(root,text='Dự đoán SVC',foreground="blue",width=20,command=forecast_SVC).place(x=70,y=350)
bt_dd_gauss=Button(root,text='Dự đoán GaussianNB',foreground="blue",width=20,command=forecast_Gauss).place(x=500,y=300)
bt_dd_id3=Button(root,text='Dự đoán ID3',foreground="blue",width=20,command=forecast_ID3).place(x=500,y=350)

label_title_ket_qua=Label(root,text="Kêt quả dự đoán",font=("Quicksand",15),bg='#FFB6C1')
label_title_ket_qua.place(x = 40 , y = 390)

tb_kq_dd=Text(root,height=7,width=105)
tb_kq_dd.place(x=40,y=430)


mainloop()