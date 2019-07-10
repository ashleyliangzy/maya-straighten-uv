
import pymel.core as pm
import pymel.core.datatypes as dt
import re
from pymel.all import *

class MB_straightenUVs():
    def __init__(self):
        print("Straighten UV Tool")
        self.errorMessage = "Points not on same edgeloop or mesh not continuous surface!"
        self.totalNumShells = 0
        self.shellsDone = 0
        if pm.window("MBUVStraightenUI", exists=True):
            deleteUI("MBUVStraightenUI")
        self.gui()

    def throwError(self, *args):
        pm.error(self.errorMessage)
        self.progressor.setProgress(0)
        self.progressor.setEnable(False)
    
    #Adding my own functions for lists because then I don't have to create additional nodes to do the work
    #And use semibroke API for Vector or Array nodes (Like Rotate)
    def add(self, *args):
        a1, a2 = zip(*args)
        return [sum(a1), sum(a2)]
    
    def sub(self, *args):
        x,y = zip(*args)
        return [reduce(lambda f,s: f-s, x), reduce(lambda f,s: f-s, y)]
    
    def multWithFloat(self, a, b):
        return [x*b for x in a]
    
    def dot(self, a, b):
        return a[0]*b[0]+a[1]*b[1]
    
    def normalize(self, vector):
        
        length = self.length(vector)
        return [vector[0]/length, vector[1]/length]
    def distance(self, a, b):
        return self.length(self.sub(a,b))
    
    def length(self, vector):
        return sqrt(vector[0]*vector[0]+vector[1]*vector[1])        
    
    def lineLineIntersect (self, line1Org, line1Dir, line2Org, line2Dir):
        
        A1 = line1Dir[1]
        B1 = -line1Dir[0]
        C1 = A1*line1Org[0]+B1*line1Org[1]
        
        A2 = line2Dir[1]
        B2 = -line2Dir[0]
        C2 = A2*line2Org[0]+B2*line2Org[1]
        
        det = A1*B2 - A2*B1
        if det == 0:
            return None
        else:
            x = (B2*C1 - B1*C2)/det
            y = (A1*C2 - A2*C1)/det
            
            
            return [x, y]
    def circularInterpolate(self, v1, v2, value):
        sin1 = asin(v1[1])
        sin2 = asin(v2[1])
        cos1 = acos(v1[0])
        cos2 = acos(v2[0])

        newSin = sin(((1-value)*sin1)+(value*sin2))
        newCos = cos(((1-value)*cos1)+(value*cos2))
        
        newPoint = [newCos, newSin]
        return newPoint    
        
        
        
    def offSetAndRotateAround(self, point, angle, pivot, offset):
        offsetPoint = self.add(point,offset)
        pivotSubstractedPoint = self.sub(offsetPoint, pivot)
        cs = cos(angle)
        sn = sin(angle)
        ex = pivotSubstractedPoint[0] * cs - pivotSubstractedPoint[1] * sn
        ye = pivotSubstractedPoint[0] * sn + pivotSubstractedPoint[1] * cs
        pivotRotatedPoint = [ex+pivot[0], ye+pivot[1]]
        return pivotRotatedPoint        
    
    def lineToPointDistance(self, a, n, p):
        aMinusP = self.sub(a,p)
        apDotN = self.dot(n,aMinusP)
        return self.distance(aMinusP,[x*apDotN for x in n])

    def SCurveRemap01(self, x, m, w, a):
        return 1/ ( 1+( a ** ( (x-0.5*m)*(w/m)) ) )    
    
    def remap(Self, x, l, h):
        return clamp((x-l)/(h-l))
    
    
    
    #Main Function
    def goThroughEdgeloopsAndStraighten(self, *args):
        #Convert and check selection
        curSelection = pm.polyListComponentConversion(tuv=True)
        curUVSelection = pm.filterExpand(curSelection, sm=35, ex=True)
        if not curUVSelection: error("Nothing Selected")
        
        #Use UV Shell index to sort points in lists by UV shell
        transformSelected = pm.listTransforms(curUVSelection[0])
        
        curUVSet = pm.polyUVSet(q=True, cuv=True)[0]
        
        UVShellPointsRelationArray = transformSelected[0].getUvShellsIds(curUVSet)
        shellSortedUVPointsArray = []
        largestShellNumber = UVShellPointsRelationArray[1]
        UVShellsList = UVShellPointsRelationArray[0]
        
        for x in range(largestShellNumber):
            shellSortedUVPointsArray.append([])
        print curUVSelection
        for point in curUVSelection:
           
            pointIndex = int(re.findall(r"[\w]+", point)[2])
            shellIndex = UVShellsList[pointIndex]
            shellSortedUVPointsArray[shellIndex].append(point)
        
        selShellInds = [x for x in range(len(shellSortedUVPointsArray)) if len(shellSortedUVPointsArray[x]) > 0]
        self.totalNumShells = len(selShellInds)
        self.progressor.setEnable(True)
            
        allShells = []
        
        #Perform straighten per shell
        for x in selShellInds:
            allShells.append(self.straighten(self.sortPoints(shellSortedUVPointsArray[x]) if self.doStraighten.getValue() else shellSortedUVPointsArray[x]))
            ++self.shellsDone
            
        #Select affected shells    
        pm.select(allShells)
        self.progressor.setEnable(False)
        self.progressor.setProgress(0)
        
        
        
        
    def sortPoints(self, startList):
        #Declarations
        checkArray = [] # Number of edges per UV
        endList = [] #Final sorted list
        endEdgeList = [] #Final edge list
        endPointsArray = [] #Indices of endpoints
        workList = [] # Edges

        #Check
        if len(startList) < 2: 
            self.throwError()
        #Convert to edges to work with
        for startListItem in startList:
             edgeList = pm.polyListComponentConversion(startListItem, fuv=True, te=True)
             workList.append(pm.filterExpand(edgeList, ex=True, sm=32))
             
             checkArray.append(0)

        #Sort list so they come in topological order

        for i in range(len(workList)):
            for j in [(k+i+1) % len(workList) for k in range(len(workList)-1)]:
                edgeShareNumberCheck = 0
                for edgeNumber in range(len(workList[j])):
                    for testEdgeNumber in range(len(workList[i])):
                       if workList[j][edgeNumber] == workList[i][testEdgeNumber]:
                           checkArray[i] += 1 
                           edgeShareNumberCheck += 1
                if edgeShareNumberCheck > 1: 
                    endPointsArray.append(i)      
        
        for i in range(len(checkArray)):
            if checkArray[i] > 3: self.throwError() 
            if checkArray[i] == 1:  endPointsArray.append(i) 
        
        
        if len(endPointsArray) != 2: self.throwError()
        startPoint = endPointsArray[0]
        endPoint = endPointsArray[1]
        
        endList.append(startList[startPoint])
        endEdgeList.append(workList[startPoint])
        
        endListEndPoint = startList[endPoint]
        del workList[endPoint]
        del startList[endPoint]
        del workList[startPoint]
        del startList[startPoint]
        
        
        
        
        while len(startList) > 0:
            foundItem = []
            for m in range(len(workList)):
                for n in range(len(workList[m])):
                    for o in range(len(endEdgeList[-1])):
                        if workList[m][n] == endEdgeList[-1][o]:
                            foundItem.append(m)
            endList.append(startList[foundItem[0]])
            endEdgeList.append(workList[foundItem[0]])
            del startList[foundItem[0]]
            del workList[foundItem[0]]
        
        endList.append(endListEndPoint)
                    
        return endList
        
    
    
    
    
    def sortShellPoints(self, inUvs, inPosList, avgLine):

        #Find UV Coordinates, Tangents and inevitably Intersectionpoints for those tangents
        tangentList = []
        posListLength = len(inPosList)   
        for distIndex in range(posListLength):
            
            curPoint = inPosList[distIndex]
            prevPoint = inPosList[clamp(distIndex-1, 0, posListLength)]
            nextPoint = inPosList[clamp(distIndex+1, 0, posListLength-1)]
            if distIndex == 0:
                tangent = self.sub(nextPoint,curPoint)
            elif distIndex != posListLength:
                tangent = self.sub(curPoint,prevPoint) 
            else: 
                tangent = self.sub(nextPoint,prevPoint)   
                
            tangent = self.normalize(tangent)
            tangentx = tangent[0]
            tangent[0] = tangent[1]
            tangent[1] = -tangentx
            tangentList.append(tangent)
        avgTangent = self.normalize(avgLine)
        avgTangentx = avgTangent[0]
        avgTangent[0] = avgTangent[1]
        avgTangent[1] = -avgTangentx  
        
             
        testPosList = []
        crossLengthList = []
        for x in range(posListLength):
            crossPoints = [self.lineLineIntersect(inPosList[x], tangentList[x], inPosList[y], tangentList[y]) for y in range(posListLength) if y != x]
            crossLengths = map(lambda pY : self.distance(pY, inPosList[x]) if pY else 10000, crossPoints)            
            crossLengthList.append(min(crossLengths))           
        
        #Go through shell points and sort by previously found tangents/distances  
        
        uvsCoordRawList = pm.polyEditUV(inUvs, q=True)
        uvsCoordList = [[uvsCoordRawList[x],uvsCoordRawList[x+1]] for x in range(0, len(uvsCoordRawList)-1, 2)]  
        uvsWeightingList = []
        for uvPoint in range(len(inUvs)):
            uvsWeightingList.append([])
            uvPos = uvsCoordList[uvPoint]
            
            closestDistance = 10000
            preClosestDistance = 10000
            postClosesDistance = 10000
            closestInd = 0
            preClosestInd = 0
            postClosestInd = 0
            
            distaList = []
            
            
            
        #Calculate Distances with point/line distance, weighting in distance between the points
            
            for li in range(posListLength):
                
                distRadius = self.distance(inPosList[li],uvPos)
                distRIFraction = clamp(distRadius/crossLengthList[li])
                tangentToUse = self.circularInterpolate(tangentList[li], avgTangent, distRIFraction)
                distLine = self.lineToPointDistance(inPosList[li], tangentToUse, uvPos)
                distaList.append(distLine+distRadius)
        #Calculate weighting of points on line based on closer needing to have largest effect    
            maxDista = max(distaList) 
            minDista = min(distaList)
            avgDista = sum(distaList)/posListLength
            weightedDistaList = map(lambda dist: (1-self.remap(dist, minDista, avgDista))**(maxDista/(minDista+0.01)), distaList)
            avgDista = sum(weightedDistaList)/posListLength
            sumDista = sum(weightedDistaList)
            
            weightedDistaList = map(lambda x: x/sumDista, weightedDistaList)
            uvsWeightingList[uvPoint] = weightedDistaList
            
                
            self.progressor.step()         
        
            
        
        return (uvsWeightingList, uvsCoordList)  
        
    
    
    def straighten(self, lineUvsList):
        #get position in lineUvsList to do this only once
        lineUvsPosList = []
        
        lineUvsRawPosList = pm.polyEditUV(lineUvsList, q=True)
        lineUvsPosList = [[lineUvsRawPosList[x],lineUvsRawPosList[x+1]] for x in range(0, len(lineUvsRawPosList)-1, 2)]
        
        
        #Find line to place points on
        directionStart = lineUvsPosList[0]
        directionEnd = lineUvsPosList[-1]
        direction = self.sub(directionEnd, directionStart)
        direction = self.normalize(direction)
        
        
        #Select part of shell not on the line, for later use
        pm.select(lineUvsList)
        mel.polySelectBorderShell(0)
        shellFullUVs = pm.filterExpand(sm=35, ex=True)
        shellUVs = [x for x in shellFullUVs if not x in lineUvsList]
        self.progressor.setMaxValue(len(shellUVs)*self.totalNumShells*2)
        self.progressor.setProgress(len(shellUVs)*self.shellsDone*2)
        
        #Are we even straightening, or just aligning?
        if self.doStraighten.getValue():
            sortedUvsArray = []
            shellUvPosList = []            
            offsetsList = []
            anglesList = []
            pivotsList = []
            
            
            #For symmetry, angles for rest of shell needs to be calculated with an offset, so separately and prior to straightening
            if self.useFishbone.getValue():
                sortedUvsAndPosList = self.sortShellPoints(shellUVs, lineUvsPosList, direction)
                shellUvWeightsList = sortedUvsAndPosList[0]
                shellUvPosList = sortedUvsAndPosList[1]
                for ind in range(len(lineUvsPosList)):
                    pre = clamp(ind-1, 0,len(lineUvsPosList)-1)
                    post = clamp(ind+1, 0, len(lineUvsPosList)-1)
                    dir = self.sub(lineUvsPosList[post],lineUvsPosList[pre])
                    angle = atan2(direction[1],direction[0]) - atan2(dir[1],dir[0])
                    anglesList.append(angle)
                    
            #Transform points on line to straight line and save offsets for shell deformation    
            oldPoint = [0,0]
            prevPoint = lineUvsPosList[1]                
            for curIndex in range(len(lineUvsList)):
                curPoint = lineUvsPosList[curIndex]
                
                translation = self.add(curPoint,oldPoint)
                if curIndex == 0:
                    offset = self.sub(prevPoint,translation)
                else:
                    offset = self.sub(translation,prevPoint)
                
                angle = atan2(direction[1],direction[0]) - atan2(offset[1],offset[0])
                CalcPos = self.offSetAndRotateAround(curPoint, angle, prevPoint, oldPoint)
                pm.polyEditUV(lineUvsList[curIndex], r=False, u=CalcPos[0], v=CalcPos[1])            
                
                if self.useFishbone.getValue():
                    offsetsList.append(self.sub(oldPoint))
                    pivotsList.append(prevPoint)
                prevPoint = CalcPos
                oldPoint=self.sub(prevPoint,curPoint)
       
            #Using strict? This is called fishbone method internally
            if self.useFishbone.getValue():
                self.fishbone(shellUVs, shellUvPosList, shellUvWeightsList, pivotsList, anglesList, offsetsList)
            #Or using old version of Unfold?
            if self.useUnfold.getValue(): 
                pm.select(shellUVs)
                pm.unfold(applyToShell=True, i=2000, us=False)
        
        #Aligning?    
        if self.alignCheck.getValue():
            self.alignUVs(directionStart, directionEnd, shellFullUVs)
        
        return shellFullUVs

    
    
    def fishbone(self, UVs, points, weights, pivots, angles, offsets):
        
        
        for shellUvIndex in range(len(UVs)):
            newPosList = [self.offSetAndRotateAround(points[shellUvIndex], angles[lineUvIndex], pivots[lineUvIndex], offsets[lineUvIndex]) for lineUvIndex in range(len(weights[shellUvIndex]))]
            shellUvsNewPosTotalList = map((lambda x,y: self.multWithFloat(x,y)), newPosList, weights[shellUvIndex])    
            shellUvsNewPosTotal = reduce(lambda x,y: self.add(x,y), shellUvsNewPosTotalList)
            pm.polyEditUV(UVs[shellUvIndex], r=False, u=shellUvsNewPosTotal[0], v=shellUvsNewPosTotal[1])
            self.progressor.step()
    
    
    
    def alignUVs(self, firstPoint, secondPoint, shellList):

        offsetPoint = [(firstPoint[0] - secondPoint[0]), (firstPoint[1] - secondPoint[1])]
        angle = dt.degrees(dt.atan2(offsetPoint[0], offsetPoint[1])) % 90
        if (dt.abs(angle) > 45): 
            angle -= 90*cmp(angle,0) 
    
        pm.polyEditUV(shellList, a=(angle), pu=(firstPoint[0]-(0.5*offsetPoint[0])), pv=(firstPoint[1]-(0.5*offsetPoint[1])))
    
        



    def gui(self): 
        
        self.win = pm.window("MBUVStraightenUI", title="Straighten UVs", rtf=True, s=False, tlb=True)
        self.layout = pm.columnLayout(adjustableColumn=True)
        self.btn = pm.button(label="Straighten", ann="Usage:\n\nSelect either edgeloops or UVs (for border edges) on an edgeloop of shells\nYou can select multiple shells\nChoose Align, Straighten or both\nSelect straighten method (Unfold is by far fastest, Strict can overwhelm the machine!)\n\nClick this button\n\nNote: For Align only, you cna select just an edge or any 2 UV points on a shell", w=200, command=self.goThroughEdgeloopsAndStraighten)
        pm.text(l="")
        self.alignCheck = pm.checkBox(v=False, label="Align Shells", ann="Use the edgeloops, post-straighten, as ruler for snapping shell to 90 degree angle")
        self.doStraighten = pm.checkBox(v=True, label="Straighten Shells", offCommand=self.StraightenOff, ann="Straighten edgeloop or just align it")
        pm.text(l="")    
        pm.text(l="Straighten Method")
        self.useUnfold = pm.checkBox(v=True, label="Use Unfold", onCommand=self.unfoldOn, ann="Use the old Unfold to straigten rest of shell (default)")
        self.useFishbone = pm.checkBox(v=False, label="Use Strict Straighten", onCommand=self.fishboneOn, offCommand=self.fishboneOff, ann="Strict straighten, can give wonky result")
        self.progressor = pm.progressBar(visible=False, enable=False, progress=0, width=200)
        self.win.show()
        
    #Change UI to fit options selected    
    def StraightenOff(self, *args):
        self.alignCheck.setValue(True)
    def fishboneOff(self, *args):
        self.progressor.setVisible(False)
        self.win.setHeight(140)
        self.win.setSizeable(False)
        self.win.setResizeToFitChildren(True)

    def fishboneOn(self, *args):
        self.useUnfold.setValue(False)
        self.progressor.setVisible(True)
        self.win.setResizeToFitChildren(True)
        self.win.setHeight(155)
        
    def unfoldOn(self, *args):
        self.useFishbone.setValue(False)
        self.fishboneOff()
MB_straightenUVs()           
