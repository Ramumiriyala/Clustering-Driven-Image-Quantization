from pathlib import Path
import joblib, cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from train_model import DATA, OUT, ART, SIZE, PATCH, STRIDE, EXTS, hog_vec, local_desc, train

st.set_page_config(page_title="Vehicle Image Intelligence",page_icon="🚘",layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.5rem}
.hero{padding:1.5rem;border-radius:18px;background:linear-gradient(120deg,#14213d,#31577e);color:white}
.hero h1{color:white;margin:0}.hero p{color:#dce8f7;margin:.4rem 0 0}
</style>
""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🚘 Vehicle Image Intelligence</h1><p>Clustering-driven image feature quantization and pattern recognition</p></div>',unsafe_allow_html=True)

def dataset_table():
    rows=[]
    if DATA.exists():
        for p in sorted(x for x in DATA.iterdir() if x.is_dir()):
            rows.append({"Vehicle class":p.name,"Images":sum(f.suffix.lower() in EXTS for f in p.rglob("*"))})
    return pd.DataFrame(rows)

def load_model():
    p=ART/"vehicle_model.joblib"
    return joblib.load(p) if p.exists() else None

def predict(upload,bundle):
    image=np.asarray(Image.open(upload).convert("RGB"))
    resized=cv2.resize(image,SIZE)
    desc=local_desc(resized)
    words=bundle["kmeans"].predict(desc)
    k=bundle["cluster_count"]
    hist=np.bincount(words,minlength=k).astype(np.float32); hist/=max(hist.sum(),1)
    label=bundle["classifier"].predict(hist.reshape(1,-1))[0]
    return image,label,hist,len(words)

model=load_model()
page=st.sidebar.radio("Navigation",["Overview","Train & Evaluate","Predict Vehicle","Explore Clusters","Dataset Guide"])
st.sidebar.markdown("---")
st.sidebar.caption("MODEL STATUS")
if model:
    st.sidebar.success(f"Trained · K={model['cluster_count']}")
    st.sidebar.caption(f"Test accuracy: {model['best_accuracy']:.1%}")
else: st.sidebar.warning("Model not trained")
st.sidebar.caption("Custom images only — no built-in dataset.")

if page=="Overview":
    st.subheader("Project dashboard")
    df=dataset_table(); total=int(df["Images"].sum()) if not df.empty else 0
    a,b,c=st.columns(3); a.metric("Dataset images",total); b.metric("Vehicle classes",len(df)); c.metric("Model","Ready" if model else "Not trained")
    st.markdown("### Dataset distribution")
    if not df.empty:
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(df,x="Vehicle class",y="Images",color="Vehicle class"),use_container_width=True)
    else: st.info("Add your images under dataset/Car, dataset/Bike, dataset/Bus and dataset/Truck.")
    st.markdown("### Workflow")
    cols=st.columns(4)
    for col,n,t,d in zip(cols,["01","02","03","04"],["Upload dataset","Extract features","Cluster features","Recognize vehicle"],["Organize photos by class","Compute local HOG descriptors","Learn K-Means visual words","Predict with an SVM"]):
        col.markdown(f"**{n} · {t}**"); col.caption(d)

elif page=="Train & Evaluate":
    st.subheader("Train model and evaluate cluster centers")
    df=dataset_table()
    if not df.empty: st.dataframe(df,use_container_width=True,hide_index=True)
    if st.button("🚀 Train / Retrain model",type="primary",use_container_width=True):
        bar=st.progress(0); msg=st.empty()
        try:
            result=train(lambda v,m:(bar.progress(v),msg.info(m)))
            st.success("Training completed.")
            st.json(result); st.rerun()
        except Exception as e:
            st.error(str(e)); st.info("Ensure each class has at least 2 readable images. For meaningful results, aim for 50–100+ images per class.")
    model=load_model()
    if model:
        x,y,z=st.columns(3); x.metric("Best quantized accuracy",f"{model['best_accuracy']:.2%}"); y.metric("Best K",model["cluster_count"]); z.metric("Global HOG baseline",f"{model['baseline_accuracy']:.2%}")
        if (OUT/"accuracy_comparison.png").exists(): st.image(str(OUT/"accuracy_comparison.png"),use_container_width=True)
        if (OUT/"confusion_matrix.png").exists(): st.image(str(OUT/"confusion_matrix.png"),use_container_width=True)
        if (OUT/"experiment_results.csv").exists(): st.dataframe(pd.read_csv(OUT/"experiment_results.csv"),use_container_width=True,hide_index=True)
        if (OUT/"classification_report.txt").exists():
            with st.expander("Classification report"): st.code((OUT/"classification_report.txt").read_text())

elif page=="Predict Vehicle":
    st.subheader("Upload an image for vehicle recognition")
    if not model: st.warning("Train the model first.")
    else:
        up=st.file_uploader("Choose a vehicle image",type=["jpg","jpeg","png","bmp","webp"])
        if up:
            try:
                image,label,hist,n=predict(up,model)
                l,r=st.columns(2)
                l.image(image,caption="Uploaded image",use_container_width=True)
                r.markdown("### Predicted category"); r.success(str(label))
                r.metric("Cluster centers",model["cluster_count"]); r.metric("Local patches",n)
                chart=pd.DataFrame({"Visual word":np.arange(len(hist)),"Frequency":hist})
                st.plotly_chart(px.bar(chart,x="Visual word",y="Frequency",title="Quantized feature histogram"),use_container_width=True)
            except Exception as e: st.error(f"Image could not be processed: {e}")

elif page=="Explore Clusters":
    st.subheader("Explore representative cluster centers")
    if not model: st.warning("Train the model first.")
    else:
        centers=model["kmeans"].cluster_centers_
        st.write(f"Learned visual vocabulary: **{len(centers)} centers**")
        k=st.slider("Select cluster center",0,len(centers)-1,0)
        st.line_chart(centers[k])
        st.caption("Each center is a representative HOG feature vector describing recurring local gradient patterns; it is not a complete vehicle image.")

else:
    st.subheader("Dataset guide")
    st.write("This project uses only images you provide. Folder names become classification labels.")
    st.code("""dataset/
  Car/
  Bike/
  Bus/
  Truck/""")
    st.markdown("""
- Add at least two classes and at least 2 readable images per class.
- Recommended: 50–100+ images per class.
- Include different viewpoints, backgrounds and lighting.
- Avoid duplicate or near-duplicate photos across train and test data.
- Supported formats: JPG, JPEG, PNG, BMP, WEBP.
""")
    df=dataset_table()
    if not df.empty: st.dataframe(df,use_container_width=True,hide_index=True)
