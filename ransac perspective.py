#ransac perspective
import numpy as np

def get_perspective(pts_src, pts_tgt):
    '''
    /* Calculates coefficients of perspective transformation
    * which maps (xi,yi) to (ui,vi), (i=1,2,3,4):
    *
    *      c00*xi + c01*yi + c02
    * ui = ---------------------
    *      c20*xi + c21*yi + c22
    *
    *      c10*xi + c11*yi + c12
    * vi = ---------------------
    *      c20*xi + c21*yi + c22
    *
    * Coefficients are calculated by solving linear system:
    * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
    * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
    * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
    * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
    * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
    * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
    * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
    * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
    *
    * where:
    *   cij - matrix coefficients, c22 = 1
    */
    '''
    assert pts_src.shape == pts_tgt.shape
    npoint = pts_src.shape[0]
    A = np.zeros((npoint*2, 8))
    b = np.zeros((npoint*2,1))
    for idx in range(npoint): 
        #|x0, y0, 1, 0, 0, 0, -x0u0, -y0u0|  -> [u0]
        #| 0, 0, 0, x0, y0, 1,-x0v0, -y0v0|  -> [v0]
        A[idx*2] = np.array([pts_src[idx, 0], pts_src[idx, 1],1,0,0,0, \
            -pts_src[idx, 0]*pts_tgt[idx, 0], -pts_src[idx, 1]*pts_tgt[idx, 0]])
        A[idx*2+1] = np.array([0, 0, 0, pts_src[idx, 0], pts_src[idx, 1], 1, \
            -pts_src[idx, 0]*pts_tgt[idx, 1], -pts_src[idx, 1]*pts_tgt[idx, 1]])
        b[idx*2] = pts_tgt[idx, 0]
        b[idx*2+1] = pts_tgt[idx, 1]
    # if pts_src.shape[0]==4: 
    X, _,_,_ = np.linalg.lstsq(A, b)
    return X
        # trans= cv2.getPerspectiveTransform(pts_src, pts_tgt)
        # print(X)
        # print(trans)
    # else: 
        
def get_inliner(pts_src, pts_tgt, face_size, X): 
    trans = np.concatenate([X.ravel(), np.ones(1)]).reshape(3,3)
    pts_src_H = np.concatenate([pts_src, np.ones((pts_src.shape[0], 1))], axis=1).T
    predict_pts = (trans@pts_src_H).T
    predict_pts[:, 0] /= predict_pts[:, 2]+1e-9
    predict_pts[:, 1] /= predict_pts[:, 2]+1e-9
    if False:
        ax = plt.subplot(1,2,1)
        plt.scatter(pts_tgt[:, 0], -pts_tgt[:, 1])
        plt.scatter(predict_pts[:, 0], -predict_pts[:, 1])
        ax = plt.subplot(1,2,2)
        plt.scatter(pts_src[:, 0], -pts_src[:, 1])
        plt.show()
    diff = np.linalg.norm(predict_pts[:, :2]-pts_tgt, axis=1)
    inliner = diff<face_size*0.03
    return inliner

def get_perspective_ransac(pts_src, pts_tgt, repeat=500):
    assert pts_src.shape == pts_tgt.shape
    npoint = pts_src.shape[0]
    best_inliner = 4
    face_size = max(pts_tgt[:, 0].max() - pts_tgt[:, 0].min(), 
                    pts_tgt[:, 1].max() - pts_tgt[:, 1].min())
    for _ in range(repeat): 
        point_sample = np.random.choice(np.arange(npoint), 4, replace=False)
        new_X = get_perspective(pts_src[point_sample, :], pts_tgt[point_sample, :])
        inliner = get_inliner(pts_src, pts_tgt, face_size, new_X)
        if np.sum(inliner) <= 4: continue 
        update_X = get_perspective(pts_src[inliner, :], pts_tgt[inliner, :])
        inliner = get_inliner(pts_src, pts_tgt, face_size, update_X)
        inliner_count = np.sum(inliner)
        if inliner_count>best_inliner: 
            best_inliner = inliner_count
            X = update_X
        # print(inliner_count)
    if best_inliner<=4: 
        return 
    else: 
        return X, best_inliner