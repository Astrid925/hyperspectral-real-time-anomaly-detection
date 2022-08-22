import os
import numpy as np
import struct
from scipy import linalg


class RAD():

    def __init__(self,filePath,gtPath,nBeginRow,nBgLine,tau):
        self.inputPath = filePath
        self.gtPath = gtPath
        self.nBeginRow = nBeginRow
        self.nBgLine = nBgLine
        self.tau = tau

    def openFile(self):
        (filePath, ext) = os.path.splitext(self.inputPath)
        (Path, filename) = os.path.split(filePath)
        os.mkdir(Path + '\\' + "ADResult_" + filename)  # 创建一个文件夹
        outPath = Path + '\\' + "ADResult_" + filename
        RXPath = os.path.join(outPath, 'RXresult.raw')
        RXheadPath = os.path.join(outPath, 'RXresult.hdr')
        self.RXFile = open(RXPath, mode='wb')
        self.RXHeadFile = open(RXheadPath, mode='w', encoding='utf-8')
        ADPath = os.path.join(outPath, 'ADresult.raw')
        ADheadPath = os.path.join(outPath, 'ADresult.hdr')
        self.ADFile = open(ADPath, mode='wb')
        self.ADHeadFile = open(ADheadPath, mode='w', encoding='utf-8')
        # 读取头文件
        headFilePath = filePath + '.hdr'
        headFile = open(headFilePath, mode='r', encoding='utf-8')
        for i in range(12):
            cLine = headFile.readline()
            if "samples" in cLine:
                strLen = len("samples")
                self.nCol = int(cLine[strLen + 3:], 10)
            elif "lines" in cLine:
                strLen = len("lines")
                self.nRow = int(cLine[strLen + 4:], 10)
            elif "bands" in cLine:
                strLen = len("bands")
                self.nBand = int(cLine[strLen + 4:], 10)
            elif "data type" in cLine:
                strLen = len("data type")
                typeValue = int(cLine[strLen + 3:], 10)
            elif "interleave" in cLine:
                strLen = len("interleave")
                fileFormat = cLine[strLen + 3:-1]
        headFile.close()
        if fileFormat != 'bil':
            print('The input file must be bil format!')
        if typeValue == 12:
            self.dataType = np.dtype(np.uint16)
            self.byteNum = 2
        elif typeValue == 4:
            self.dataType = np.dtype(np.float32)
            self.byteNum = 4
        elif typeValue == 5:
            self.dataType = np.dtype(np.float64)
            self.byteNum = 8

    def bytesToValue(self, byteList, byteNum):
        Len = len(byteList) // byteNum
        data = np.zeros((1, Len), dtype=self.dataType, order='C')
        if byteNum == 4:
            for i in range(Len):
                value = struct.unpack('<f', byteList[i * byteNum:i * byteNum + byteNum])[0]
                data[:, i] = value
        elif byteNum == 8:
            for i in range(Len):
                value = struct.unpack('<d', byteList[i * byteNum:i * byteNum + byteNum])[0]
                data[:, i] = value
        elif byteNum == 2:
            for i in range(Len):
                value = struct.unpack('<H', byteList[i * byteNum:i * byteNum + byteNum])[0]
                data[:, i] = value
        elif byteNum == 1:
            for i in range(Len):
                value = struct.unpack('<?', byteList[i * byteNum:i * byteNum + byteNum])[0]
                data[:, i] = value
        return data

    def RXStart(self, bgM, pR, nBeginRow, nCol):
        result = np.zeros((nBeginRow, nCol), dtype=np.float64, order='C')
        row, col = np.diag_indices_from(pR)
        pR[row, col] = np.diagonal(pR, offset=0) + 0.000001
        pRinv = np.linalg.inv(pR)
        for i in range(nBeginRow):
            for j in range(nCol):
                result[i, j] = np.matmul(np.matmul(bgM[i, j, :], pRinv), np.transpose(bgM[i, j, :]))
        return result

    def lineRXresult(self, Linv, lineData, nCol):
        Lineresult = np.zeros((1, nCol), dtype=np.float64, order='C')
        for i in range(nCol):
            temp = np.matmul(Linv, lineData[:, i])
            Lineresult[:, i] = np.linalg.norm(temp)
        return Lineresult

    def targetDetection(self):
        openFile = open(self.inputPath, mode='rb')
        fileSize = os.path.getsize(self.inputPath)
        rowCount = 0
        startBg = np.zeros((self.nBeginRow, self.nCol, self.nBand), dtype=self.dataType, order='C')  # 数据类型要自动
        self.pRXresult = np.zeros((self.nRow, self.nCol), dtype=np.float64, order='C')
        pR = np.zeros((self.nBand, self.nBand), dtype=np.float64, order='C')
        pRSet = np.zeros((self.nBgLine, self.nBand, self.nBand), dtype=np.float64, order='C')
        while True:
            if openFile.tell() > fileSize - 1:
                print("已到文件末尾!")
                RXtext = "ENVI\n" + "samples = " + str(self.nCol) + "\n" + "lines = " + str(self.nRow) + "\n" + "bands = 1\n" + "header offset = 0\n" + "file type = ENVI Standard\n" + "data type = 5\n" + "interleave = bsq\n" + "byte order = 0"
                self.RXFile.write(self.pRXresult.astype('<d'))
                self.RXHeadFile.write(RXtext)
                openFile.close()
                self.RXFile.close()
                self.RXHeadFile.close()
                break
            else:
                temp = openFile.read(self.byteNum * self.nCol * self.nBand)
                lineData = self.bytesToValue(temp, self.byteNum)
                frameData = lineData.reshape((self.nBand, self.nCol))
                lineR = 1 / self.nCol * np.matmul(frameData, np.transpose(frameData))
                if rowCount < self.nBeginRow:
                    startBg[rowCount, :] = np.transpose(frameData)
                    pR = (1 - 1 / (rowCount + 1)) * pR + 1 / (rowCount + 1) * lineR
                    pRSet[rowCount, :] = pR
                    # if rowCount == self.nBeginRow - 1:
                    #     self.pRXresult[0:self.nBeginRow, :] = self.RXStart(startBg, pR, self.nBeginRow, self.nCol)
                elif self.nBeginRow <= rowCount < self.nBgLine:
                    pR = (1 - 1 / (rowCount + 1)) * pR + 1 / (rowCount + 1) * lineR
                    L = linalg.cholesky(pR, lower=True)
                    Linv = np.linalg.inv(L)
                    self.pRXresult[rowCount, :] = self.lineRXresult(Linv, frameData, self.nCol)
                    pRSet[rowCount, :] = pR
                else:
                    pR = pR + 1 / self.nBgLine * (lineR - pRSet[0])
                    L = linalg.cholesky(pR, lower=True)
                    Linv = np.linalg.inv(L)
                    self.pRXresult[rowCount, :] = self.lineRXresult(Linv, frameData, self.nCol)
                    pRSet[0:self.nBgLine - 1, :] = pRSet[1:, :]
                    pRSet[-1, :] = lineR
                rowCount += 1
                print(str(rowCount))

    def resultEvaluation(self):
        gtFile = open(self.gtPath, mode='rb')
        gt = gtFile.read(self.nRow * self.nCol)
        Map = self.bytesToValue(gt, 1)
        Map = np.reshape(Map, (self.nRow, self.nCol))
        posNum = np.count_nonzero(Map)
        negNum = np.sum(Map == 0)
        RXresult = (self.pRXresult - self.pRXresult.min()) / (self.pRXresult.max() - self.pRXresult.min())
        pos = np.argwhere(RXresult > self.tau)
        gtValue = Map[pos[:, 0], pos[:, 1]]
        TP = np.count_nonzero(gtValue)
        FP = np.sum(gtValue == 0)
        PD = TP / posNum
        PF = FP / negNum
        print("探测率：" + str(PD))
        print("虚警率：" + str(PF))
        # 写探测文件
        RXresult[RXresult <= self.tau] = 0
        RXresult[RXresult > self.tau] = 1
        RXresult = RXresult.reshape((self.nRow, self.nCol))
        ADtext = "ENVI\n" + "samples =" + str(self.nCol) + "\n" + "lines=" + str(self.nRow) + "\n" + "bands =1\n" + "header offset = 0\n" + "file type = ENVI Standard\n" + "data type = 1\n" + "interleave = bsq\n" + "byte order = 0"
        self.ADFile.write(RXresult.astype('<?'))
        self.ADHeadFile.write(ADtext)
        self.ADFile.close()
        self.ADHeadFile.close()


if __name__ == "__main__":
    filePath = r'G:\Dataset\reflectionData\flat_ac\region\08_vnir\08_vnir_ac_bil'
    gtPath = r'G:\Dataset\reflectionData\flat_ac\region\08_vnir\08_vnir_roi'
    nBeginRow = 65# 起始行
    nBgLine = 220  # 背景行
    tau = 0.1 # 探测阈值
    myTest = RAD(filePath, gtPath, nBeginRow, nBgLine,tau)
    myTest.openFile()
    myTest.targetDetection()
    myTest.resultEvaluation()
    # myTest.enhanceResult()

