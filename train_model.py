from pathlib import Path
import json, csv
import cv2, joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE=Path(__file__).resolve().parent
DATA=BASE/"dataset"; OUT=BASE/"outputs"; ART=BASE/"artifacts"
SIZE=(128,128); PATCH=32; STRIDE=16
KS=[8,16,32,64]; EXTS={".jpg",".jpeg",".png",".bmp",".webp"}

def read_data():
    xs=[]; ys=[]
    DATA.mkdir(parents=True,exist_ok=True)
    for folder_name in ("Car","Bike","Bus","Truck"):
        (DATA/folder_name).mkdir(parents=True,exist_ok=True)
    folders=sorted(p for p in DATA.iterdir() if p.is_dir())
    if len(folders)<2: raise ValueError("Add at least two class folders under dataset/.")
    for folder in folders:
        n=0
        for f in folder.rglob("*"):
            if f.suffix.lower() not in EXTS: continue
            im=cv2.imread(str(f))
            if im is None: continue
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            xs.append(cv2.resize(im,SIZE)); ys.append(folder.name); n+=1
        print(f"{folder.name}: {n} images")
    ys=np.asarray(ys)
    names,counts=np.unique(ys,return_counts=True)
    if len(xs)==0 or len(names)<2 or np.any(counts<2):
        raise ValueError("Need at least two classes, with at least two readable images in each.")
    return np.asarray(xs),ys,names

def hog_vec(gray):
    return hog(gray,orientations=9,pixels_per_cell=(8,8),
               cells_per_block=(2,2),block_norm="L2-Hys",
               feature_vector=True).astype(np.float32)

def local_desc(im):
    gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY); out=[]
    for y in range(0,gray.shape[0]-PATCH+1,STRIDE):
        for x in range(0,gray.shape[1]-PATCH+1,STRIDE):
            out.append(hog_vec(gray[y:y+PATCH,x:x+PATCH]))
    return np.asarray(out,dtype=np.float32)

def per_image_desc(images):
    per=[local_desc(im) for im in images]
    return per,np.vstack(per)

def histograms(per,km,k):
    rows=[]
    for d in per:
        h=np.bincount(km.predict(d),minlength=k).astype(np.float32)
        h/=max(h.sum(),1); rows.append(h)
    return np.asarray(rows)

def global_features(images):
    return np.asarray([hog_vec(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)) for im in images])

def train(progress=None):
    OUT.mkdir(exist_ok=True); ART.mkdir(exist_ok=True)
    images,labels,names=read_data()
    ids=np.arange(len(labels))
    tr,te=train_test_split(ids,test_size=.25,random_state=42,stratify=labels)
    xtr,xte=images[tr],images[te]; ytr,yte=labels[tr],labels[te]
    if progress: progress(.08,"Extracting local HOG descriptors...")
    train_per,all_desc=per_image_desc(xtr); test_per,_=per_image_desc(xte)
    base=make_pipeline(StandardScaler(),SVC(kernel="rbf",C=5))
    base.fit(global_features(xtr),ytr)
    base_acc=accuracy_score(yte,base.predict(global_features(xte)))
    if len(all_desc)>20000:
        rng=np.random.default_rng(42)
        all_desc=all_desc[rng.choice(len(all_desc),20000,replace=False)]
    valid=[k for k in KS if k<=len(all_desc)]
    if not valid: raise ValueError("Not enough training image patches for clustering.")
    results=[]; best=None
    for i,k in enumerate(valid):
        if progress: progress(.15+.72*i/len(valid),f"Training K-Means with {k} centers...")
        km=KMeans(n_clusters=k,random_state=42,n_init=10).fit(all_desc)
        a=histograms(train_per,km,k); b=histograms(test_per,km,k)
        clf=make_pipeline(StandardScaler(),SVC(kernel="rbf",C=5))
        clf.fit(a,ytr); pred=clf.predict(b); acc=accuracy_score(yte,pred)
        results.append({"cluster_centers":k,"test_accuracy":float(acc)})
        if best is None or acc>best["accuracy"]:
            best={"accuracy":float(acc),"k":k,"km":km,"clf":clf,"pred":pred}
    joblib.dump({"kmeans":best["km"],"classifier":best["clf"],
        "cluster_count":best["k"],"class_names":list(names),
        "baseline_accuracy":float(base_acc),"best_accuracy":best["accuracy"],
        "results":results},ART/"vehicle_model.joblib")
    with (OUT/"experiment_results.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["cluster_centers","test_accuracy"]); w.writeheader(); w.writerows(results)
    ks=[r["cluster_centers"] for r in results]; scores=[r["test_accuracy"] for r in results]
    plt.figure(figsize=(8,5)); plt.plot(ks,scores,marker="o",label="Quantized local HOG")
    plt.axhline(base_acc,ls="--",label=f"Global HOG baseline ({base_acc:.3f})")
    plt.xlabel("Number of cluster centers"); plt.ylabel("Test accuracy")
    plt.title("Vehicle recognition vs cluster-center count"); plt.xticks(ks)
    plt.ylim(0,1.02); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(OUT/"accuracy_comparison.png",dpi=160); plt.close()
    ConfusionMatrixDisplay.from_predictions(yte,best["pred"],labels=names,
        display_labels=names,xticks_rotation=45,cmap="Blues",colorbar=False)
    plt.tight_layout(); plt.savefig(OUT/"confusion_matrix.png",dpi=160); plt.close()
    report=classification_report(yte,best["pred"],labels=names,target_names=names,zero_division=0)
    (OUT/"classification_report.txt").write_text(report,encoding="utf-8")
    summary={"total_images":len(images),"training_images":len(xtr),"testing_images":len(xte),
      "classes":list(names),"baseline_accuracy":float(base_acc),
      "best_cluster_count":best["k"],"best_accuracy":best["accuracy"],"results":results}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if progress: progress(1.0,"Training complete. Model and reports saved.")
    return summary

if __name__=="__main__":
    try:
        print(json.dumps(train(),indent=2))
    except ValueError as exc:
        raise SystemExit(str(exc))
