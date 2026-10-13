"""Train MobileNetV2 on a directory of class folders without split leakage.
Expected dataset layout: data/trashnet/<class_name>/*.jpg
"""
import argparse, json, random
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def main():
    parser=argparse.ArgumentParser(description="Train EcoVision MobileNetV2")
    parser.add_argument("--data", default="data/trashnet", help="class-folder dataset directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args=parser.parse_args(); random.seed(args.seed); np.random.seed(args.seed)
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc: raise SystemExit("TensorFlow is required. Install requirements.txt first.") from exc
    data_dir=Path(args.data); class_names=sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if not class_names: raise SystemExit(f"No class folders found in {data_dir}")
    extensions={".jpg",".jpeg",".png",".webp"}; items=[]
    for label,name in enumerate(class_names):
        for path in (data_dir/name).rglob("*"):
            if path.suffix.lower() in extensions: items.append((str(path), label))
    if len(items)<len(class_names)*3: raise SystemExit("Each class needs at least three readable images.")
    paths, labels=zip(*items); paths=np.array(paths); labels=np.array(labels)
    train_p, hold_p, train_y, hold_y=train_test_split(paths, labels, test_size=.30, random_state=args.seed, stratify=labels)
    val_p, test_p, val_y, test_y=train_test_split(hold_p, hold_y, test_size=2/3, random_state=args.seed, stratify=hold_y)
    def ds(p,y,shuffle):
        d=tf.data.Dataset.from_tensor_slices((p,y))
        def load(path,label):
            image=tf.io.read_file(path); image=tf.image.decode_image(image, channels=3, expand_animations=False); image=tf.image.resize(image,[224,224]); return tf.cast(image,tf.float32)/255.0,label
        d=d.map(load,num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle: d=d.shuffle(len(p),seed=args.seed)
        return d.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    aug=keras.Sequential([layers.RandomFlip("horizontal"),layers.RandomRotation(.08),layers.RandomZoom(.1)],name="augmentation")
    base=keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights="imagenet"); base.trainable=False
    inputs=keras.Input((224,224,3)); x=aug(inputs); x=base(x,training=False); x=layers.GlobalAveragePooling2D()(x); x=layers.Dropout(.25)(x); outputs=layers.Dense(len(class_names),activation="softmax")(x)
    model=keras.Model(inputs,outputs); model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    model.fit(ds(train_p,train_y,True),validation_data=ds(val_p,val_y,False),epochs=args.epochs)
    probs=model.predict(ds(test_p,test_y,False),verbose=0); pred=probs.argmax(1)
    report=classification_report(test_y,pred,target_names=class_names,output_dict=True,zero_division=0)
    out=Path("models"); out.mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)
    model.save(out/"ecovision_mobilenetv2.keras"); json.dump({str(i):n for i,n in enumerate(class_names)},open(out/"labels.json","w"),indent=2)
    json.dump({"dataset":"TrashNet-derived directory","seed":args.seed,"counts":{"train":len(train_p),"validation":len(val_p),"test":len(test_p)},"accuracy":report["accuracy"],"per_class":{n:report[n] for n in class_names}},open("reports/metrics.json","w"),indent=2)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,6)); ax.imshow(confusion_matrix(test_y,pred),cmap="Greens"); ax.set(xticks=range(len(class_names)),yticks=range(len(class_names)),xticklabels=class_names,yticklabels=class_names,xlabel="Predicted",ylabel="Actual",title="EcoVision AI test confusion matrix"); plt.setp(ax.get_xticklabels(),rotation=35,ha="right"); fig.tight_layout(); fig.savefig("reports/confusion_matrix.png",dpi=160); print(json.dumps({"accuracy":report["accuracy"],"classes":class_names},indent=2))
if __name__=="__main__": main()
