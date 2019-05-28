import numpy as np
import cv2, os

blue = (255,0,0)
red = (0,0,255)
green = (0,255,0)

def bound(img,p1,p2,p3,p4,color,line_width=3):
    p1 = (int(p1[0]),int(p1[1]))
    p2 = (int(p2[0]),int(p2[1]))
    p3 = (int(p3[0]),int(p3[1]))
    p4 = (int(p4[0]),int(p4[1]))

    cv2.line(img,p1,p2,green,1)
    cv2.line(img,p2,p3,green,1)
    cv2.line(img,p3,p4,green,1)
    cv2.line(img,p4,p1,green,1)

    x,y,w,h = cv2.boundingRect(np.array([[p1,p2,p3,p4]]))
    cv2.rectangle(img,(x,y),(x+w,y+h),color,line_width)

def get_grad(img,template,w,tl):
    grad_x = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, ksize = 5)
    grad_y = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, ksize = 5)

    warp_grad_x = cv2.warpAffine(grad_x, w, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    warp_grad_y = cv2.warpAffine(grad_y, w, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    
    crop_grad_x = warp_grad_x[tl[1]:tl[1]+template.shape[0], tl[0]:tl[0]+template.shape[1]]
    crop_grad_y = warp_grad_y[tl[1]:tl[1]+template.shape[0], tl[0]:tl[0]+template.shape[1]]

    return crop_grad_x,crop_grad_y

def get_error(img,template,w,tl,light=False):
   
    warp_img = cv2.warpAffine(img, w, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        
    crop_img = warp_img[tl[1]:tl[1]+template.shape[0], tl[0]:tl[0]+template.shape[1]]

    if light:
        temp_mean = np.mean(template)
        warp_mean = np.mean(crop_img)
        diff = temp_mean/warp_mean
        crop_img = crop_img*diff

    error = (template.astype(int)-crop_img.astype(int))
    error = error.reshape(-1,1)

    return error

def steepest(template,grad_x,grad_y):
    delta = np.zeros((template.shape[0]*template.shape[1],6))
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            I_px = np.array([grad_x[i,j],grad_y[i,j]])
            jac_px = np.array([[j,0,i,0,1,0],[0,j,0,i,0,1]])
            delta[i*template.shape[1]+j] = np.dot(I_px,jac_px).reshape(1,-1)
    return delta

def hessian(img,template,warp,tl):
    H = np.zeros((6,6))

    grad_x,grad_y = get_grad(img,template,warp,tl)
    delta = steepest(template,grad_x,grad_y)

    H = np.matmul(delta.T,delta)

    return H,delta

def LK(img, template, rect, p, alpha, thresh, light):
    
    warp = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])

    tl = rect[0]
    tr = rect[1]
    br = rect[2]
    bl = rect[3]
    
    norm = 100
    threshold = thresh
    count = 0

    while norm > threshold and count < 15:
        # print(norm)
        error = get_error(img,template,warp,tl,light)
        grad_x,grad_y = get_grad(img,template,warp,tl)
        H,delta = hessian(img,template,warp,tl)
        dp = np.matmul(np.matmul(np.linalg.pinv(H),delta.T),error)
        
        norm = np.linalg.norm(dp)
        
        dp = alpha*dp
   
        p[0] += dp[0]
        p[1] += dp[1]
        p[2] += dp[2]
        p[3] += dp[3]
        p[4] += dp[4]
        p[5] += dp[5]
            
        warp = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])

        count += 1
    
    matrix = np.vstack([warp,[0,0,1]])
    
    p1 = np.matmul(matrix,np.array([tl[0], tl[1], 1]).reshape(1,-1).T)
    p2 = np.matmul(matrix,np.array([tr[0], tr[1], 1]).reshape(1,-1).T)
    p3 = np.matmul(matrix,np.array([br[0], br[1], 1]).reshape(1,-1).T)
    p4 = np.matmul(matrix,np.array([bl[0], bl[1], 1]).reshape(1,-1).T)

    p1 = (int(p1[0]),int(p1[1]))
    p2 = (int(p2[0]),int(p2[1]))
    p3 = (int(p3[0]),int(p3[1]))
    p4 = (int(p4[0]),int(p4[1]))

    new_points = [p1,p2,p3,p4]
        
    return p, new_points
    
def main(top_left,width,height,start=0,folder=None,use_frame=False,alpha=100,thresh=0.006,light=False):
    image_folder = folder

    frame = cv2.imread(image_folder+('frame' if use_frame else '')+str(start).zfill(4)+'.jpg')
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    tl = top_left
    w,h = width,height

    p1 = (tl[0],tl[1])
    p2 = (tl[0]+w,tl[1])
    p3 = (tl[0]+w,tl[1]+h)
    p4 = (tl[0],tl[1]+h)

    points = [p1,p2,p3,p4]

    template = frame[tl[1]:tl[1]+h , tl[0]:tl[0]+w]
    p = np.zeros(6) 

    o1 = p1
    o2 = p2
    o3 = p3
    o4 = p4

    for i in range(len(images)):
        img = cv2.imread(image_folder+('frame' if use_frame else '')+str(start+i).zfill(4)+'.jpg')

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        p, new_points = LK(gray,template,points,p,alpha,thresh,light)
        p1,p2,p3,p4 = new_points

        # bound(img,o1,o2,o3,o4,blue)
        bound(img,p1,p2,p3,p4,red)

        cv2.imshow('frame', img)
        # cv2.imwrite('video_box_bounding/frame_'+str(i)+'.png',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# CAR SEQUENCE
main((125,105),210,170,start=20,folder='data/car/',use_frame=True,light=True)

# HUMAN SEQUENCE
# main((253,290),35,75,start=140,folder='data/human/',use_frame=False,alpha=3,thresh=0.05)

# VASE SEQUENCE
# main((120,88),55,65,start=19,folder='data/vase/',use_frame=False,alpha=120,thresh =.0005)