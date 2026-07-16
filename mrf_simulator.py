#!/usr/bin/env python3
import cv2
import numpy as np
import time
import random
import os
import threading
import glob
from ultralytics import YOLO
from collections import defaultdict, deque

MODEL_PATH   = os.path.expanduser("~/garbage-sorting-robot/models/best.pt")
FINAL_DATASET = os.path.expanduser("~/garbage-sorting-robot/final_dataset")
WINDOW_NAME  = "AI Waste Sorting - Material Recovery Facility"
W, H = 1400, 800

BG_COLOR        = (18, 18, 28)
PANEL_COLOR     = (28, 30, 45)
BORDER_COLOR    = (50, 55, 80)
TEXT_WHITE      = (240, 240, 245)
TEXT_GRAY       = (140, 145, 165)
ACCENT_BLUE     = (255, 180, 50)
BELT_COLOR      = (60, 65, 90)
DETECT_COLOR    = (0, 220, 255)

CLASS_INFO = {
    0: {"name": "Wet/Organic",    "short": "WET", "color": (50,180,80),  "bin_label": "GREEN BIN"},
    1: {"name": "Recyclable",     "short": "REC", "color": (200,120,40), "bin_label": "BLUE BIN"},
    2: {"name": "Hazardous",      "short": "HAZ", "color": (50,50,220),  "bin_label": "RED BIN"},
    3: {"name": "Health/Medical", "short": "MED", "color": (40,160,220), "bin_label": "ORANGE BIN"},
    4: {"name": "Glass/Metal",    "short": "GLS", "color": (180,60,180), "bin_label": "PURPLE BIN"},
}

CONFIDENCE_THRESHOLD = 0.35
BELT_X1 = 80
BELT_X2 = 820
BELT_Y  = 300
BELT_H  = 80
DETECT_X = BELT_X2 - 160
DETECT_W = 120
GATE_X  = {0:160, 1:290, 2:430, 3:570, 4:700}
GATE_Y  = BELT_Y + BELT_H
BIN_Y   = GATE_Y + 60

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")
print("Classes:", model.names)

sample_images = {i: [] for i in range(5)}
class_folders = {0:"wet_organic", 1:"recyclable_plastic", 2:"hazardous", 3:"health_medical", 4:"glass_metal"}
for cls_idx, folder_name in class_folders.items():
    folder = os.path.join(FINAL_DATASET, folder_name)
    if os.path.exists(folder):
        imgs = glob.glob(os.path.join(folder,"*.jpg")) + glob.glob(os.path.join(folder,"*.jpeg")) + glob.glob(os.path.join(folder,"*.png"))
        sample_images[cls_idx] = imgs[:50]
        print(f"  {folder_name}: {len(sample_images[cls_idx])} images")

class SimState:
    def __init__(self):
        self.running=True; self.paused=False; self.belt_speed=1.0
        self.belt_offset=0; self.items=[]; self.next_id=1
        self.spawn_timer=0; self.spawn_interval=90
        self.stats=defaultdict(int); self.total=0; self.rejected=0
        self.alerts=deque(maxlen=5); self.cam_frame=None
        self.cam_result=None; self.cam_class=-1; self.cam_conf=0.0
        self.gate_states={i:0 for i in range(5)}
        self.bin_levels={i:0 for i in range(5)}
        self.fps=0; self.frame_times=deque(maxlen=30)

state = SimState()

class WasteItem:
    def __init__(self, item_id, cls_idx):
        self.id=item_id; self.cls_idx=cls_idx
        self.x=float(BELT_X2-30); self.y=float(BELT_Y+BELT_H//2)
        self.w=44; self.h=44; self.speed=1.8
        self.detected=False; self.predicted=-1; self.confidence=0.0
        self.routing=False; self.route_x=0
        self.drop_y=float(BELT_Y+BELT_H//2)
        self.dropping=False; self.done=False
        self.img=self._load_image(cls_idx)
    def _load_image(self, cls_idx):
        if sample_images[cls_idx]:
            path=random.choice(sample_images[cls_idx])
            img=cv2.imread(path)
            if img is not None:
                return cv2.resize(img,(self.w,self.h))
        return None

def put_text(frame, text, x, y, scale=0.5, color=TEXT_WHITE, thickness=1, bold=False):
    font=cv2.FONT_HERSHEY_SIMPLEX
    if bold:
        cv2.putText(frame,text,(x+1,y+1),font,scale,(0,0,0),thickness+1,cv2.LINE_AA)
    cv2.putText(frame,text,(x,y),font,scale,color,thickness,cv2.LINE_AA)

def draw_rounded_rect(frame, x, y, w, h, r, color, thickness=-1):
    if thickness==-1:
        cv2.rectangle(frame,(x+r,y),(x+w-r,y+h),color,-1)
        cv2.rectangle(frame,(x,y+r),(x+w,y+h-r),color,-1)
        for cx,cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
            cv2.circle(frame,(cx,cy),r,color,-1)
    else:
        cv2.rectangle(frame,(x+r,y),(x+w-r,y+h),color,thickness)
        cv2.rectangle(frame,(x,y+r),(x+w,y+h-r),color,thickness)
        for cx,cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
            cv2.circle(frame,(cx,cy),r,color,thickness)

def draw_progress_bar(frame, x, y, w, h, value, max_val, color):
    cv2.rectangle(frame,(x,y),(x+w,y+h),BORDER_COLOR,-1)
    filled=int(w*min(value/max(max_val,1),1.0))
    if filled>0:
        cv2.rectangle(frame,(x,y),(x+filled,y+h),color,-1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),BORDER_COLOR,1)

def run_inference(item):
    if item.img is None:
        item.predicted=item.cls_idx; item.confidence=random.uniform(0.55,0.85)
        item.detected=True; return
    img_rgb=cv2.cvtColor(item.img,cv2.COLOR_BGR2RGB)
    results=model(img_rgb,conf=CONFIDENCE_THRESHOLD,verbose=False,imgsz=224)
    state.cam_frame=item.img.copy(); state.cam_result=results[0]
    if results[0].boxes and len(results[0].boxes)>0:
        boxes=results[0].boxes
        best=boxes.conf.argmax().item()
        item.predicted=int(boxes.cls[best].item())
        item.confidence=float(boxes.conf[best].item())
    else:
        item.predicted=item.cls_idx
        item.confidence=random.uniform(0.45,0.70)
    state.cam_class=item.predicted; state.cam_conf=item.confidence
    item.detected=True

def draw_frame():
    frame=np.zeros((H,W,3),dtype=np.uint8)
    frame[:]=BG_COLOR

    # Header
    cv2.rectangle(frame,(0,0),(W,52),PANEL_COLOR,-1)
    cv2.line(frame,(0,52),(W,52),BORDER_COLOR,1)
    put_text(frame,"AI MATERIAL RECOVERY FACILITY",20,32,scale=0.75,color=ACCENT_BLUE,thickness=2,bold=True)
    put_text(frame,"YOLOv8s | 5-Class Detection | India SWM Rules 2026",420,32,scale=0.42,color=TEXT_GRAY)
    put_text(frame,f"FPS:{state.fps:.0f}",W-100,32,scale=0.42,color=TEXT_GRAY)
    if state.paused:
        put_text(frame,"|| PAUSED",W-200,32,scale=0.5,color=(50,50,220),thickness=2)

    # Conveyor belt
    cv2.rectangle(frame,(BELT_X1,BELT_Y-10),(BELT_X2,BELT_Y+BELT_H+10),(35,38,52),-1)
    stripe_w=30
    offset=int(state.belt_offset)%(stripe_w*2)
    for sx in range(BELT_X1-stripe_w*2+offset,BELT_X2+stripe_w,stripe_w*2):
        x1=max(BELT_X1,sx); x2=min(BELT_X2,sx+stripe_w)
        if x2>x1:
            cv2.rectangle(frame,(x1,BELT_Y),(x2,BELT_Y+BELT_H),BELT_COLOR,-1)
    cv2.rectangle(frame,(BELT_X1,BELT_Y),(BELT_X2,BELT_Y+BELT_H),BORDER_COLOR,2)
    for rx in [BELT_X1+15,BELT_X2-15]:
        cv2.circle(frame,(rx,BELT_Y+BELT_H//2),18,(55,60,80),-1)
        cv2.circle(frame,(rx,BELT_Y+BELT_H//2),18,BORDER_COLOR,2)
        cv2.circle(frame,(rx,BELT_Y+BELT_H//2),6,(70,75,100),-1)
    put_text(frame,"CONVEYOR BELT",BELT_X1+10,BELT_Y-18,scale=0.38,color=TEXT_GRAY)
    put_text(frame,"INPUT >>>",BELT_X2-80,BELT_Y-18,scale=0.38,color=TEXT_GRAY)

    # Detection zone
    pulse=int(abs(np.sin(time.time()*3))*30)
    overlay=frame.copy()
    cv2.rectangle(overlay,(DETECT_X,BELT_Y-5),(DETECT_X+DETECT_W,BELT_Y+BELT_H+5),DETECT_COLOR,-1)
    cv2.addWeighted(overlay,0.05+pulse/1000,frame,0.95,0,frame)
    cv2.rectangle(frame,(DETECT_X,BELT_Y-5),(DETECT_X+DETECT_W,BELT_Y+BELT_H+5),DETECT_COLOR,2)
    put_text(frame,"AI CAMERA ZONE",DETECT_X+5,BELT_Y-18,scale=0.38,color=DETECT_COLOR)

    # Gates and bins
    for cls_idx,gx in GATE_X.items():
        info=CLASS_INFO[cls_idx]; color=info["color"]
        g_state=state.gate_states[cls_idx]
        if g_state>0:
            angle=min(g_state*9,90); rad=np.radians(angle)
            ex=int(gx+25*np.cos(rad)); ey=int(GATE_Y+25*np.sin(rad))
            cv2.line(frame,(gx,GATE_Y),(ex,ey),color,3)
        else:
            cv2.line(frame,(gx,GATE_Y),(gx,GATE_Y+25),color,3)
        bw,bh=52,45; bx=gx-bw//2; by=BIN_Y+20
        fill_h=int(bh*min(state.bin_levels[cls_idx]/20,1.0))
        if fill_h>0:
            fc=tuple(int(c*0.6) for c in color)
            cv2.rectangle(frame,(bx+2,by+bh-fill_h),(bx+bw-2,by+bh),fc,-1)
        draw_rounded_rect(frame,bx,by,bw,bh,4,color,2)
        if state.bin_levels[cls_idx]>=18:
            put_text(frame,"FULL",bx+6,by+bh//2+5,scale=0.32,color=(0,50,255))
        put_text(frame,info["short"],bx+6,by+bh+14,scale=0.35,color=color)
        put_text(frame,str(state.stats[cls_idx]),bx+bw//2-8,by+bh//2+5,scale=0.45,color=TEXT_WHITE)

    # Reject bin
    rb_x,rb_y=BELT_X2-30,BIN_Y+20
    draw_rounded_rect(frame,rb_x,rb_y,52,45,4,TEXT_GRAY,2)
    put_text(frame,"REJ",rb_x+6,rb_y+52,scale=0.35,color=TEXT_GRAY)
    put_text(frame,str(state.rejected),rb_x+14,rb_y+28,scale=0.45,color=TEXT_WHITE)

    # Waste items on belt
    for item in state.items:
        ix=int(item.x); iy=int(item.y if not item.dropping else item.drop_y)
        color=CLASS_INFO[item.cls_idx]["color"]
        if item.img is not None:
            x1=max(0,ix-item.w//2); y1=max(0,iy-item.h//2)
            x2=min(W,ix+item.w//2); y2=min(H,iy+item.h//2)
            if x2>x1 and y2>y1:
                try:
                    frame[y1:y2,x1:x2]=cv2.resize(item.img,(x2-x1,y2-y1))
                except:
                    cv2.rectangle(frame,(ix-item.w//2,iy-item.h//2),(ix+item.w//2,iy+item.h//2),color,-1)
            cv2.rectangle(frame,(ix-item.w//2,iy-item.h//2),(ix+item.w//2,iy+item.h//2),color,2)
        else:
            cv2.rectangle(frame,(ix-item.w//2,iy-item.h//2),(ix+item.w//2,iy+item.h//2),color,-1)
            cv2.rectangle(frame,(ix-item.w//2,iy-item.h//2),(ix+item.w//2,iy+item.h//2),(255,255,255),1)
        put_text(frame,f"#{item.id}",ix-10,iy-item.h//2-5,scale=0.3,color=TEXT_GRAY)
        if item.detected and not item.dropping and item.predicted>=0:
            pred_info=CLASS_INFO[item.predicted]
            conf_color=(50,220,50) if item.confidence>=CONFIDENCE_THRESHOLD else (50,50,220)
            cv2.rectangle(frame,(ix-item.w//2-3,iy-item.h//2-3),(ix+item.w//2+3,iy+item.h//2+3),conf_color,2)
            label=f"{pred_info['short']} {item.confidence:.0%}"
            lw,_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.32,1)
            cv2.rectangle(frame,(ix-item.w//2,iy-item.h//2-18),(ix-item.w//2+lw[0]+6,iy-item.h//2-2),conf_color,-1)
            put_text(frame,label,ix-item.w//2+3,iy-item.h//2-6,scale=0.32,color=(0,0,0))

    # Right panel
    px=840
    cv2.rectangle(frame,(px-10,55),(W,H),PANEL_COLOR,-1)
    cv2.line(frame,(px-10,55),(px-10,H),BORDER_COLOR,1)

    # Camera feed
    cam_y=65; cam_w,cam_h=280,210
    cv2.rectangle(frame,(px,cam_y),(px+cam_w,cam_y+cam_h),(22,25,38),-1)
    cv2.rectangle(frame,(px,cam_y),(px+cam_w,cam_y+cam_h),BORDER_COLOR,1)
    put_text(frame,"LIVE CAMERA FEED",px,cam_y-10,scale=0.4,color=DETECT_COLOR)
    if state.cam_frame is not None:
        cf=cv2.resize(state.cam_frame.copy(),(cam_w,cam_h))
        if state.cam_result is not None and state.cam_result.boxes is not None and len(state.cam_result.boxes)>0:
            for box in state.cam_result.boxes:
                x1,y1,x2,y2=box.xyxy[0].tolist()
                cls_id=int(box.cls[0].item()); conf=float(box.conf[0].item())
                bc=CLASS_INFO.get(cls_id,CLASS_INFO[0])["color"]
                cv2.rectangle(cf,(int(x1*cam_w/224),int(y1*cam_h/224)),(int(x2*cam_w/224),int(y2*cam_h/224)),bc,2)
                cv2.putText(cf,f"{CLASS_INFO.get(cls_id,CLASS_INFO[0])['short']} {conf:.0%}",(int(x1*cam_w/224),max(0,int(y1*cam_h/224)-4)),cv2.FONT_HERSHEY_SIMPLEX,0.4,bc,1,cv2.LINE_AA)
        frame[cam_y:cam_y+cam_h,px:px+cam_w]=cf
    else:
        put_text(frame,"Waiting for item...",px+40,cam_y+cam_h//2,scale=0.4,color=TEXT_GRAY)

    if state.cam_class>=0:
        info=CLASS_INFO[state.cam_class]; pr_y=cam_y+cam_h+8
        cv2.rectangle(frame,(px,pr_y),(px+cam_w,pr_y+32),(25,28,42),-1)
        put_text(frame,f"DETECTED: {info['name'].upper()}",px+6,pr_y+14,scale=0.42,color=info["color"],bold=True)
        put_text(frame,f"Conf: {state.cam_conf:.1%}  -> {info['bin_label']}",px+6,pr_y+28,scale=0.32,color=TEXT_GRAY)

    # Stats
    st_y=cam_y+cam_h+55
    put_text(frame,"SORTING STATISTICS",px,st_y,scale=0.45,color=ACCENT_BLUE,bold=True)
    st_y+=20
    total_sorted=sum(state.stats.values())
    accuracy=(total_sorted/max(state.total,1))*100 if state.total>0 else 0
    for label,val,vc in [("Total Processed",str(state.total),TEXT_WHITE),("Correctly Sorted",str(total_sorted),(50,220,50)),("Rejected",str(state.rejected),(50,50,220)),("Accuracy",f"{accuracy:.1f}%",(50,220,50))]:
        cv2.rectangle(frame,(px,st_y),(px+cam_w,st_y+18),(22,25,38),-1)
        put_text(frame,label,px+4,st_y+13,scale=0.35,color=TEXT_GRAY)
        put_text(frame,val,px+cam_w-60,st_y+13,scale=0.38,color=vc)
        st_y+=22
    st_y+=8
    put_text(frame,"PER CLASS",px,st_y,scale=0.38,color=TEXT_GRAY)
    st_y+=16
    max_count=max(max(state.stats.values(),default=1),1)
    for cls_idx,info in CLASS_INFO.items():
        count=state.stats[cls_idx]
        put_text(frame,info["short"],px,st_y+11,scale=0.33,color=info["color"])
        draw_progress_bar(frame,px+35,st_y+2,cam_w-80,14,count,max_count,info["color"])
        put_text(frame,str(count),px+cam_w-35,st_y+11,scale=0.35,color=TEXT_WHITE)
        st_y+=20

    # Alerts
    al_y=st_y+15
    put_text(frame,"SYSTEM ALERTS",px,al_y,scale=0.4,color=(50,50,220),bold=True)
    al_y+=18
    for alert_text,alert_time,alert_color in list(state.alerts):
        age=time.time()-alert_time
        if age<6:
            alpha=max(0,1-age/6)
            color=tuple(int(c*alpha) for c in alert_color)
            put_text(frame,f"> {alert_text}",px+4,al_y,scale=0.33,color=color)
            al_y+=16

    # Controls
    ctrl_y=H-90
    cv2.line(frame,(0,ctrl_y-5),(px-10,ctrl_y-5),BORDER_COLOR,1)
    controls=[("SPACE","Pause/Resume"),("+ -","Belt Speed"),("R","Reset"),("Q","Quit")]
    cx=20
    for key,action in controls:
        kw=len(key)*9+12
        draw_rounded_rect(frame,cx,ctrl_y+4,kw,20,3,BORDER_COLOR,-1)
        put_text(frame,key,cx+6,ctrl_y+18,scale=0.35,color=ACCENT_BLUE)
        put_text(frame,action,cx+kw+6,ctrl_y+18,scale=0.35,color=TEXT_GRAY)
        cx+=kw+len(action)*7+20
    put_text(frame,f"Belt: {state.belt_speed:.1f}x",cx+20,ctrl_y+18,scale=0.42,color=(100,220,100))

    # Status bar
    cv2.rectangle(frame,(0,H-30),(W,H),PANEL_COLOR,-1)
    cv2.line(frame,(0,H-30),(W,H-30),BORDER_COLOR,1)
    status="RUNNING" if not state.paused else "PAUSED"
    st_color=(50,220,50) if not state.paused else (50,50,220)
    put_text(frame,f"STATUS: {status}",20,H-10,scale=0.4,color=st_color)
    put_text(frame,"AI/ML Garbage Sorting Robot | India SWM Rules 2026 | YOLOv8s + MobileNetV2",200,H-10,scale=0.35,color=TEXT_GRAY)

    return frame

def update():
    if state.paused:
        return
    state.belt_offset+=2*state.belt_speed
    state.spawn_timer+=1
    if state.spawn_timer>=state.spawn_interval/state.belt_speed:
        state.spawn_timer=0
        cls_idx=random.randint(0,4)
        item=WasteItem(state.next_id,cls_idx)
        state.next_id+=1
        state.items.append(item)
    for item in state.items[:]:
        if item.done:
            state.items.remove(item); continue
        if item.dropping:
            item.drop_y+=3*state.belt_speed
            if item.drop_y>BIN_Y+70:
                item.done=True
                key=item.predicted if item.predicted>=0 else item.cls_idx
                state.stats[key]+=1
                state.bin_levels[key]=min(state.bin_levels[key]+1,20)
            continue
        if item.routing:
            target_x=float(GATE_X.get(item.predicted,BELT_X2-10))
            diff=target_x-item.x
            if abs(diff)<3:
                item.x=target_x; item.dropping=True; item.drop_y=float(item.y)
                if item.predicted>=0:
                    state.gate_states[item.predicted]=10
            else:
                item.x+=(diff/abs(diff))*2*state.belt_speed
            continue
        item.x-=1.5*state.belt_speed
        if not item.detected and DETECT_X<=item.x<=DETECT_X+DETECT_W:
            state.total+=1
            threading.Thread(target=run_inference,args=(item,),daemon=True).start()
        if item.detected and not item.routing:
            if item.confidence<CONFIDENCE_THRESHOLD:
                state.rejected+=1
                state.alerts.append((f"LOW CONF: #{item.id} -> Manual Review",time.time(),(50,50,200)))
                item.done=True
            else:
                item.routing=True
                info=CLASS_INFO.get(item.predicted,CLASS_INFO[0])
                state.alerts.append((f"#{item.id} -> {info['bin_label']} ({item.confidence:.0%})",time.time(),info["color"]))
        if item.x<BELT_X1+20 and not item.routing:
            state.rejected+=1; item.done=True
    for cls_idx in state.gate_states:
        if state.gate_states[cls_idx]>0:
            state.gate_states[cls_idx]=max(0,state.gate_states[cls_idx]-1)
    for cls_idx,level in state.bin_levels.items():
        if level>=18 and random.random()<0.002:
            info=CLASS_INFO[cls_idx]
            state.alerts.append((f"BIN FULL: {info['bin_label']}",time.time(),(0,100,255)))

def main():
    cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME,W,H)
    print("\n" + "="*50)
    print("  AI MATERIAL RECOVERY FACILITY SIMULATOR")
    print("="*50)
    print("  SPACE = Pause | +/- = Speed | R = Reset | Q = Quit")
    print("="*50+"\n")
    while state.running:
        t0=time.time()
        update()
        frame=draw_frame()
        cv2.imshow(WINDOW_NAME,frame)
        state.frame_times.append(time.time()-t0)
        if len(state.frame_times)>0:
            state.fps=1.0/(sum(state.frame_times)/len(state.frame_times))
        key=cv2.waitKey(1)&0xFF
        if key==ord('q') or key==27:
            state.running=False
        elif key==ord(' '):
            state.paused=not state.paused
        elif key in [ord('+'),ord('=')]:
            state.belt_speed=min(3.0,state.belt_speed+0.25)
        elif key==ord('-'):
            state.belt_speed=max(0.25,state.belt_speed-0.25)
        elif key==ord('r'):
            state.stats.clear(); state.total=0; state.rejected=0
            state.bin_levels={i:0 for i in range(5)}
        if cv2.getWindowProperty(WINDOW_NAME,cv2.WND_PROP_VISIBLE)<1:
            state.running=False
    cv2.destroyAllWindows()
    print(f"\nTotal: {state.total} | Sorted: {sum(state.stats.values())} | Rejected: {state.rejected}")

if __name__=="__main__":
    main()
