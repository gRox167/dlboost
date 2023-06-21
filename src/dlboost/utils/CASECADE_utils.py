import os
import pydicom
import glob
import numpy as np
import xml.etree.ElementTree as ET

def readDicom(path):
    pi = os.path.basename(path).split('_')[1]
    dcm_size = len(glob.glob(path+'/*.dcm'))
    dcms = [path+'/E'+pi+'S101I%d.dcm'%dicom_slicei for dicom_slicei in range(1,dcm_size+1)]
    dcm_f = pydicom.read_file(dcms[0]).pixel_array
    dcm_size = max(dcm_f.shape)
    dcm_img = np.zeros((dcm_size,dcm_size,len(dcms)))
    for dcmi in range(len(dcms)):
        cdcm = pydicom.read_file(dcms[dcmi]).pixel_array
        dcm_img[dcm_size//2-cdcm.shape[0]//2:dcm_size//2+cdcm.shape[0]//2,
                dcm_size//2-cdcm.shape[1]//2:dcm_size//2+cdcm.shape[1]//2,dcmi] = cdcm
    return dcm_img

def listContourSlices(qvsroot):
    avail_slices = []
    qvasimg = qvsroot.findall('QVAS_Image')
    for dicom_slicei in range(dcm_img.shape[2]):
        conts = qvasimg[dicom_slicei - 1].findall('QVAS_Contour')
        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices

def getContour(qvsroot,dicomslicei,conttype,dcmsz=720):
    qvasimg = qvsroot.findall('QVAS_Image')
    if dicomslicei - 1 > len(qvasimg):
        print('no slice', dicomslicei)
        return
    assert int(qvasimg[dicomslicei - 1].get('ImageName').split('I')[-1]) == dicomslicei
    conts = qvasimg[dicomslicei - 1].findall('QVAS_Contour')
    tconti = -1
    for conti in range(len(conts)):
        if conts[conti].find('ContourType').text == conttype:
            tconti = conti
            break
    if tconti == -1:
        print('no such contour', conttype)
        return
    pts = conts[tconti].find('Contour_Point').findall('Point')
    contours = []
    for pti in pts:
        contx = float(pti.get('x')) / 512 * dcmsz 
        conty = float(pti.get('y')) / 512 * dcmsz 
        #if current pt is different from last pt, add to contours
        if len(contours) == 0 or contours[-1][0] != contx or contours[-1][1] != conty:
            contours.append([contx, conty])
    return np.array(contours)

def removeAllContours(qvsroot, seriesnum = '101'):
    #find qvas_image
    qvasimgs = qvsroot.findall('QVAS_Image')
   
    # clear all previous contours 
    for imgi in range(len(qvasimgs)):
        
        cts = qvasimgs[imgi].findall('QVAS_Contour')
        for ctsi in cts:
            print('deleting contour in ', qvasimgs[imgi].get('ImageName') )
            qvasimgs[imgi].remove(ctsi)
                   
                    
def setContour(qvsroot, tslicei, contour, contype, dcmsz=720, contour_conf = None, seriesnum = '101', cont_comment = None):
    #qvsroot: root xml node
    #tslicei: dicom slice number corresponding to the contour you want to set
    #contour: contour coordinates in [x,y] array
    #contype: "Lumen" or "Outerwall", if previous contour of same contype exists, it will be deleted
    #dcmsz: x dimension of dicom image

    #find qvas_image
    qvasimgs = qvsroot.findall('QVAS_Image')
    fdqvasimg = -1
    for slicei in range(len(qvasimgs)):
        if qvasimgs[slicei].get('ImageName').split('S'+seriesnum+'I')[-1] == str(tslicei):
            fdqvasimg = slicei
            break
    if fdqvasimg == -1:
        print('QVAS_IMAGE not found')
        return

    # clear previous contours if there are
    cts = qvasimgs[fdqvasimg].findall('QVAS_Contour')
    for ctsi in cts:
        ctype = ctsi.findall('ContourType')
        for ctr in ctype:
            if ctr.text == contype:
                qvasimgs[fdqvasimg].remove(ctsi)

    #add new contour
    if contype == "Outer Wall":
        ct = "Outer Wall"
        ctcl = '16776960'
    elif contype == "Lumen":
        ct = "Lumen"
        ctcl = '255'

    QVAS_Contour = ET.SubElement(qvasimgs[fdqvasimg], 'QVAS_Contour')            

    Contour_Point = ET.SubElement(QVAS_Contour, 'Contour_Point')
    for coutnodei in contour:
        Point = ET.SubElement(Contour_Point, 'Point')
        Point.set('x', '%.5f'%(coutnodei[0] / dcmsz * 512))
        Point.set('y', '%.5f'%(coutnodei[1] / dcmsz * 512))

    ContourType = ET.SubElement(QVAS_Contour, 'ContourType')
    ContourType.text = ct

    ContourColor = ET.SubElement(QVAS_Contour, 'ContourColor')
    ContourColor.text = ctcl

    #-------------Ignore below, only for software loading purposes ----------------
    ContourOpenStatus = ET.SubElement(QVAS_Contour, 'ContourOpenStatus')
    ContourOpenStatus.text = '1'
    ContourPCConic = ET.SubElement(QVAS_Contour, 'ContourPCConic')
    ContourPCConic.text = '0.5'
    ContourSmooth = ET.SubElement(QVAS_Contour, 'ContourSmooth')
    ContourSmooth.text = '60'
    Snake_Point = ET.SubElement(QVAS_Contour, 'Snake_Point')
    #snake point, fake fill
    for snakei in range(6):
        conti = len(contour)//6*snakei
        Point = ET.SubElement(Snake_Point, 'Point')
        Point.set('x', '%.5f'%(contour[conti][0] / dcmsz * 512))
        Point.set('y', '%.5f'%(contour[conti][1] / dcmsz * 512))

    ContourComments = ET.SubElement(QVAS_Contour, 'ContourComments')
    if cont_comment is not None:
        ContourComments.text = cont_comment
    #-------------Ignore above, only for software loading purposes ----------------

    ContourConf = ET.SubElement(QVAS_Contour, 'ContourConf')
    if contour_conf is not None:
        LumenConsistency = ET.SubElement(ContourConf, 'LumenConsistency')
        LumenConsistency.text = '%.5f'%contour_conf[0]
        WallConsistency = ET.SubElement(ContourConf, 'WallConsistency')
        WallConsistency.text = '%.5f'%contour_conf[1]