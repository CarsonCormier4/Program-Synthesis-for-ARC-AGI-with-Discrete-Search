from collections import deque
import heapq
import json
import time

# Grammar representation
class Program:
    def __init__(self, op=None, left=None, right=None):
        self.op = op  # Operation type
        self.left = left  # For Sequence: left program
        self.right = right  # For Sequence: right program or args
        if op and left and right:
            self.complexity = left.complexity + right.complexity
        elif op:
            self.complexity = 1
        else:
            self.complexity = 0

    def __str__(self):
        if self.op == 'Sequence':
            left_op = self.left.op if self.left else 'None'
            right_op = self.right.op if self.right else 'None'
            return f"Sequence({left_op}, {right_op})"
        elif self.op == 'ColorChange':
            return f"ColorChange({self.right[0]}, {self.right[1]})"
        elif self.op == 'Mirror':
            return f"Mirror({self.right})"
        elif self.op == 'Rotate':
            return f"Rotate({self.right})"
        elif self.op == 'Scale2x2':
            return "Scale2x2()"
        elif self.op == 'Scale3x3':
            return "Scale3x3()"
        elif self.op == 'Scale2x1':
            return "Scale2x1()"
        elif self.op == 'Scale1x2':
            return "Scale1x2()"
        elif self.op == 'ResizeIrregular':
            return f"ResizeIrregular({self.right[0]}x{self.right[1]})"
        elif self.op == 'PositionalShift':
            return f"PositionalShift({self.right[0]}, {self.right[1]}, {self.right[2]}, {self.right[3]})"
        elif self.op == 'ColorMapMultiple':
            return f"ColorMapMultiple({dict(self.right)})"
        elif self.op == 'ScaleWithColorMap':
            return f"ScaleWithColorMap({self.right[0]}, {dict(self.right[1])})"
        elif self.op == 'SwapColors':
            return f"SwapColors({self.right[0]}, {self.right[1]})"
        elif self.op == 'DiagonalReflection':
            return f"DiagonalReflection({self.right[0]}, {self.right[1]})"
        return ""

    def __lt__(self, other):
        # For heap comparison, use complexity as tiebreaker
        return self.complexity < other.complexity

    def __eq__(self, other):
        return (self.op == other.op and
                self.left == other.left and
                self.right == other.right)



################################
# Task 1: Program Application  #
################################
def apply_program(input_grid, program):
    """Apply a program to an input grid and return the output grid."""
    if program is None:
        return None

    grid = [row[:] for row in input_grid]  # Deep copy

    if program.op == 'ColorChange':
        old_color, new_color = program.right
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == old_color:
                    grid[i][j] = new_color

    elif program.op == 'Mirror':
        axis = program.right
        if axis == 'horizontal':
            grid = grid[::-1]  # Flip vertically
        elif axis == 'vertical':
            grid = [row[::-1] for row in grid]  # Flip horizontally

    elif program.op == 'Rotate':
        degrees = program.right
        if degrees == 90:
            grid = [[grid[len(grid)-1-j][i] for j in range(len(grid))]
                    for i in range(len(grid[0]))]

        # TODO: Implement 180 and 270 degree rotation
        elif degrees==180:
            #G'[i][j] = G[m-1-i][n-1-j]
            grid=[[grid[len(grid)-1-i][len(grid[0])-1-j]
                     for j in range(len(grid[0]))]
                    for i in range(len(grid))]
        elif degrees==270:
            #G'[i][j] = G[j][n-1-i]
            grid=[[grid[j][len(grid[0])-1-i]
                     for j in range(len(grid))]
                    for i in range(len(grid[0]))]

    #Scale2x2 algorithm
    elif program.op == 'Scale2x2':
        # TODO: Implement 2x2 scaling
        newGrid=[]
        for i in range(len(grid)):
            row1,row2= [],[]
            for j in range(len(grid[0])):
                #Append G[i][j] twice to row1 and row2
                row1.extend([grid[i][j]]*2)
                row2.extend([grid[i][j]]*2)
            newGrid.append(row1)
            newGrid.append(row2)
        grid=newGrid

    #Scale3x3 algorithm
    elif program.op == 'Scale3x3':
        # TODO: Implement 3x3 scaling
        newGrid=[]
        for i in range(len(grid)):
            row1,row2,row3=[],[],[]
            for j in range(len(grid[0])):
                #Append G[i][j] three times to each row
                row1.extend([grid[i][j]]*3)
                row2.extend([grid[i][j]]*3)
                row3.extend([grid[i][j]]*3)
            newGrid.extend([row1,row2,row3])
        grid=newGrid

    # TODO: Implement other operations (Scale2x1, Scale1x2, PositionalShift, etc.)
    #Scale2x1 algorithm
    elif program.op == 'Scale2x1':
        newGrid=[]
        for i in range(len(grid)):
            newRow=[]
            for j in range(len(grid[0])):
                #Append G[i][j] twice to new row
                newRow.extend([grid[i][j]]*2)
            newGrid.append(newRow)
        grid= newGrid

    #Scale1x2 algorithm
    elif program.op == 'Scale1x2':
        newGrid=[]
        for i in range(len(grid)):
            #Append copy of G[i] to G' twice
            newGrid.append(grid[i][:])
            newGrid.append(grid[i][:])
        grid=newGrid

    #ResizeIrregular algorithm
    elif program.op == 'ResizeIrregular':
        newHeight,newWidth=program.right
        newGrid=[]
        for i in range(newHeight):
            newRow=[]
            for j in range(newWidth):
                #orig_i ← min(i, m-1); orig_j ← min(j, n-1)
                origI=min(i,len(grid)-1)
                origJ=min(j,len(grid[0])-1)
                newRow.append(grid[origI][origJ])
            newGrid.append(newRow)
        grid=newGrid

    #PositionalShift algorithm
    elif program.op=='PositionalShift':
        oldColor,newColor,dr,dc=program.right
        newGrid=[row[:] for row in grid]

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c]== oldColor:
                    #G'[r][c] ← 0 (Clear original position)
                    newGrid[r][c]=0
                    #new_r ← r + dr; new_c ← c + dc
                    newR=r+dr
                    newC=c+dc
                    #If 0 ≤ new_r < m and 0 ≤ new_c < n then
                    if 0<=newR<len(grid) and 0<=newC<len(grid[0]):
                        newGrid[newR][newC]=newColor
        grid=newGrid

    #ColorMapMultiple algorithm
    elif program.op=='ColorMapMultiple':
        colorMap=dict(program.right)
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                #If G'[r][c] ∈ M then G'[r][c] ← M[G'[r][c]]
                if grid[i][j] in colorMap:
                    grid[i][j]=colorMap[grid[i][j]]

    #ScaleWithColorMap algorithm
    elif program.op=='ScaleWithColorMap':
        s, mapping=program.right
        mapping=dict(mapping)
        newGrid=[]
        for i in range(len(grid)):
            #For k = 0 to s-1 (create s rows for each original row)
            for _ in range(s):
                newRow=[]
                for j in range(len(grid[i])):
                    color=grid[i][j]
                    #mapped_color ← M[G[i][j]] if G[i][j] ∈ M else G[i][j]
                    mappedColor=mapping[color] if color in mapping else color
                    #Repeat s times horizontally
                    newRow.extend([mappedColor]*s)
                newGrid.append(newRow)
        grid=newGrid

    #SwapColors algorithm
    elif program.op=='SwapColors':
        c1,c2=program.right
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j]==c1:
                    grid[i][j]=c2
                elif grid[i][j]==c2:
                    grid[i][j]=c1

    #DiagonalReflection algorithm
    elif program.op=='DiagonalReflection':
        oldColor, newColor=program.right
        newGrid=[row[:] for row in grid]
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c]==oldColor:
                    newGrid[r][c] = 0
                    if c<len(newGrid) and r<len(newGrid[0]):
                        newGrid[c][r]=newColor
        grid=newGrid

    #Sequence algorithm
    elif program.op=='Sequence':
        #Apply left program first, then right program
        grid=apply_program(grid, program.left)
        grid=apply_program(grid, program.right)

    return grid


#Gets all unique colors from the training data
def getColorsFromData(trainData):
    colors=set()
    for inp, out in trainData:
        for row in inp:
            colors.update(row)
        if out:
            for row in out:
                colors.update(row)
    return list(colors)


def generateBasicOperations(trainData):
    """Generate basic operations dynamically based on training data."""
    colors=getColorsFromData(trainData)
    ops=[]

    #Basic operations
    ops.extend([
        Program('Mirror',None,'horizontal'),
        Program('Mirror',None,'vertical'),
        Program('Rotate',None,90),
        Program('Rotate',None,180),
        Program('Rotate',None,270),
        Program('Scale2x2',None,None),
        Program('Scale3x3',None,None),
        Program('Scale2x1',None,None),
        Program('Scale1x2',None,None),
        Program('ResizeIrregular',None,[2, 2]),
    ])

    #Color operations with common colors
    commonColors=[0,1,2,3,4,5]
    for c1 in commonColors:
        for c2 in commonColors:
            if c1!=c2:
                ops.append(Program('ColorChange',None,[c1, c2]))
                ops.append(Program('SwapColors',None,[c1, c2]))
                ops.append(Program('ColorMapMultiple',None,[(c1, c2)]))
                ops.append(Program('ScaleWithColorMap', None,[2, [(c1, c2)]]))
                ops.append(Program('DiagonalReflection',None,[c1, c2]))

    #Add positional shifts
    directions = [(0,1),(1,0),(-1,0),(0,-1)]
    for c1 in [1,2]:
        for c2 in [2,3]:
            for dr, dc in directions:
                ops.append(Program('PositionalShift',None,[c1,c2,dr,dc]))

    return ops


################################
#   Task 2: BFS Search         #
################################
def bfs_search(train_data, max_complexity):
    """BFS search for program synthesis."""
    # TODO: Generate basic operations and add to queue
    # TODO: Implement main BFS loop
    # TODO: Check programs for correctness
    # TODO: Generate new programs by sequencing

    queue=deque()
    visited=set()

    #Generate basic operations
    basicOperations=generateBasicOperations(train_data)

    #Initialize queue
    for op in basicOperations:
        queue.append(op)
        visited.add(str(op))

    explored=0
    maxExplored=50000

    #Main BFS loop
    while queue and explored<maxExplored:
        explored+=1
        program=queue.popleft()

        #Check if program produces the correct output
        allCorrect=True
        for inGrid, outGrid in train_data:
            result=apply_program(inGrid, program)
            if result!=outGrid:
                allCorrect=False
                break

        if allCorrect:
            return program

        #Generate new programs by sequencing
        if program.complexity+1<=max_complexity:
            for op in basicOperations:
                newProg=Program('Sequence',program,op)
                programStr=str(newProg)
                if programStr not in visited and newProg.complexity<=max_complexity:
                    visited.add(programStr)
                    queue.append(newProg)

    return None  # Return found program or None


################################
#   Task 3: GBFS Search        #
################################
def gbfs_search(train_data, max_complexity, heuristic_fn):
    """Greedy Best-First Search for program synthesis using the provided heuristic function."""
    # TODO: Generate basic operations and add to heap with h-values only
    # TODO: Implement main GBFS loop
    # TODO: Use heuristic function to compute priorities (ignore path cost)
    # TODO: Generate new programs and maintain heap invariant

    heap=[]
    visited=set()

    #Generate basic operations
    basicOperations=generateBasicOperations(train_data)

    #Add basic operations to the heap with h-values only
    for op in basicOperations:
        hVal=heuristic_fn(op, train_data)
        heapq.heappush(heap,(hVal,id(op),op))
        visited.add(str(op))

    explored=0
    max_explored=50000

    #Main GBFS loop
    while heap and explored<max_explored:
        explored+=1
        #Use heuristic function to compute priorities (ignore path cost)
        hVal,_,program=heapq.heappop(heap)

        #Check if program works
        works=True
        for inGrid, outGrid in train_data:
            result=apply_program(inGrid,program)
            if result!=outGrid:
                works=False
                break

        if works:
            return program

        #Generate new programs
        if program.complexity+1<=max_complexity:
            for op in basicOperations:
                newProg=Program('Sequence',program,op)
                programStr=str(newProg)
                if programStr not in visited and newProg.complexity<=max_complexity:
                    visited.add(programStr)
                    hVal=heuristic_fn(newProg,train_data)
                    heapq.heappush(heap,(hVal,id(newProg),newProg))

    return None  # Return found program or None


################################
#   Task 4: A* Search          #
################################
def a_star_search(train_data, max_complexity, heuristic_fn):
    """A* search for program synthesis using the provided heuristic function."""
    # TODO: Generate basic operations and add to heap with f-values
    # TODO: Implement main A* loop
    # TODO: Use heuristic function to compute priorities
    # TODO: Generate new programs and maintain heap invariant

    heap=[]
    visited=set()

    #Generate basic operations
    basicOperations=generateBasicOperations(train_data)

    #Add basic operations to heap with f-values
    for op in basicOperations:
        gVal=1
        hVal=heuristic_fn(op,train_data)
        fVal=gVal+hVal
        heapq.heappush(heap,(fVal,id(op),op,gVal))
        visited.add(str(op))

    explored=0
    maxExplored=50000

    #Main A* loop
    while heap and explored<maxExplored:
        explored+=1

        #Compute f(n) = g(n) + h(n) using heuristic function
        fVal,_,program,gVal=heapq.heappop(heap)

        #Check if program works
        allCorrect=True
        for inGrid,outGrid in train_data:
            result=apply_program(inGrid,program)
            if result!=outGrid:
                allCorrect=False
                break

        if allCorrect:
            return program

        #Generate new programs
        if program.complexity+1<=max_complexity:
            for op in basicOperations:
                newProg=Program('Sequence',program,op)
                programStr=str(newProg)
                if programStr not in visited and newProg.complexity<=max_complexity:
                    visited.add(programStr)
                    newGValue=gVal+1
                    newHValue=heuristic_fn(newProg,train_data)
                    newFValue=newGValue+newHValue
                    heapq.heappush(heap,(newFValue,id(newProg),newProg,newGValue))

    return None  # Return found program or None


#Heuristic Function (for GBFS and A*)
def basicHeuristic(program,train_data):
    """Basic cell mismatch heuristic."""
    totalMismatches=0
    for inGrid,outGrid in train_data:
        result=apply_program(inGrid,program)
        if result is None:
            totalMismatches +=1000
            continue

        #Compare overlapping cells
        rows=min(len(result),len(outGrid))
        cols=min(len(result[0]) if result else 0, len(outGrid[0]) if outGrid else 0)

        #Count mismatched cells
        for i in range(rows):
            for j in range(cols):
                if result[i][j]!=outGrid[i][j]:
                    totalMismatches+=1

        #Penalize size differences
        totalMismatches+=abs(len(result)-len(outGrid))*10
        if result and outGrid:
            totalMismatches+=abs(len(result[0])-len(outGrid[0]))*10

    return totalMismatches


################################
#   Task 5: Custom Heuristics  #
################################
def heuristic_custom_1(program,train_data):
    """Your first custom heuristic function."""
    # TODO: Design a creative and effective heuristic
    # Consider: output shape, color distributions, symmetries, etc.

    heuristicVal=0
    for inGrid,outGrid in train_data:
        result=apply_program(inGrid,program)
        if result is None:
            heuristicVal+=1000
            continue

        #Count cell mismatches
        rows = min(len(result),len(outGrid))
        cols = min(len(result[0]) if result else 0,len(outGrid[0]) if outGrid else 0)
        for i in range(rows):
            for j in range(cols):
                if result[i][j]!=outGrid[i][j]:
                    heuristicVal+=2

        #Big penalty for size mismatches
        heuristicVal+=abs(len(result)-len(outGrid))*20
        if result and outGrid:
            heuristicVal+=abs(len(result[0])-len(outGrid[0]))*20

        #Color distribution analysis
        def countColors(grid):
            counts={}
            for row in grid:
                for color in row:
                    counts[color]=counts.get(color,0)+1
            return counts

        resultColors=countColors(result)
        targetColors=countColors(outGrid)

        allColors=set(resultColors.keys()) | set(targetColors.keys())
        for color in allColors:
            countDiff=abs(resultColors.get(color,0)-targetColors.get(color,0))
            heuristicVal+=countDiff*3

    return heuristicVal


def heuristic_custom_2(program,train_data):
    """Second custom heuristic focusing on structural patterns."""
    heuristicVal=0
    for inGrid,outGrid in train_data:
        result=apply_program(inGrid,program)
        if result is None:
            heuristicVal+=1000
            continue

        #Basic cell matching
        rows=min(len(result),len(outGrid))
        cols=min(len(result[0]) if result else 0,len(outGrid[0]) if outGrid else 0)

        #Count mismatched cells
        for i in range(rows):
            for j in range(cols):
                if result[i][j]!=outGrid[i][j]:
                    heuristicVal+=1

        #Enhanced dimension penalties
        heuristicVal+=abs(len(result)-len(outGrid))*15
        if result and outGrid:
            heuristicVal+=abs(len(result[0])-len(outGrid[0]))*15

        #Border color analysis
        if result and outGrid:

            #Check if border colors match
            resultBorders=set()
            targetBorders=set()

            #Top and bottom borders
            for j in range(len(result[0])):
                resultBorders.add(result[0][j])
                resultBorders.add(result[-1][j])
            for j in range(len(outGrid[0])):
                targetBorders.add(outGrid[0][j])
                targetBorders.add(outGrid[-1][j])

            #Left and right borders
            for i in range(len(result)):
                resultBorders.add(result[i][0])
                resultBorders.add(result[i][-1])
            for i in range(len(outGrid)):
                targetBorders.add(outGrid[i][0])
                targetBorders.add(outGrid[i][-1])

            borderDiff=len(resultBorders.symmetric_difference(targetBorders))
            heuristicVal+=borderDiff*5

    return heuristicVal


################################
#          Test Runner         #
################################
def runTests(searchMethods=['bfs','gbfs','astar'],maxComplexity=5):
    """Run tests on ARC challenges."""

    #Load challenge and solution data from JSON files
    try:
        with open("arc-agi_challenges.json") as f:
            challenges=json.load(f)
        with open("arc-agi_solutions.json") as f:
            solutions=json.load(f)

    #Exit if the files are missing
    except FileNotFoundError:
        print("Required JSON files not found.")
        return

    #Clean the data
    for taskID,taskData in challenges.items():
        for example in taskData.get("train",[]):

            #Keep only input and output keys
            keysToRemove=[key for key in example.keys() if key not in ("input","output")]
            for key in keysToRemove:
                example.pop(key)

    #Map search method names to their corresponding function(s)
    methodMap={
        'bfs':(bfs_search, None),
        'gbfs':(gbfs_search, basicHeuristic),
        'astar':(a_star_search, basicHeuristic),
        'gbfs_custom1':(gbfs_search, heuristic_custom_1),
        'astar_custom1':(a_star_search, heuristic_custom_1),
        'gbfs_custom2':(gbfs_search, heuristic_custom_2),
        'astar_custom2':(a_star_search, heuristic_custom_2)
    }

    #Accuracy counters for each method
    accuracies={method:0 for method in searchMethods}
    totalTasks=len(challenges)

    #Loop through each ARC challenge
    for taskID,taskData in challenges.items():
        print(f"\n=== Testing Task {taskID} ===")
        trainExamples=[(ex["input"], ex["output"]) for ex in taskData["train"]]

        #Skip if there aren't any solutions for the challenge
        if taskID not in solutions:
            print("No solution available for this task.")
            continue

        #Initialize test inpur and the expected output
        testInput=taskData["test"][0]["input"]
        expectedOutput=solutions[taskID][0]

        #Run each search method on the current task
        for method in searchMethods:
            if method not in methodMap:
                continue

            searchFN,heuristic_fn=methodMap[method]
            startTime=time.time()
            try:

                #Run the search function (Use heuristic if needed)
                if heuristic_fn is not None:
                    program=searchFN(trainExamples,maxComplexity,heuristic_fn)
                else:
                    program=searchFN(trainExamples,maxComplexity)
            except Exception as e:
                print(f"Error in {method}: {e}")
                program=None
            elapsed=time.time()-startTime

            #Format the method label properly
            methodLabel=method.upper()
            if 'custom' in method:
                methodLabel+=f" (Custom Heuristic {method[-1]})"

            #If a valid program is found, test it
            if program is not None:
                print(f"{methodLabel} Program: {program}, Time: {elapsed:.2f}s")
                testOutput=apply_program(testInput,program)
                print(f"Test Output: {testOutput}")

                #Count as correct if output matches the expected one
                if testOutput==expectedOutput:
                    accuracies[method]+=1
            else:

                #Display if the search failed
                print(f"{methodLabel} Program: *****No Program Found*****")

        print(f"Expected Output: {expectedOutput}")
        print("\n" + "="*30)

    #Final accuracy summary
    print("\n===============================")
    print("ACCURACY SUMMARY")
    print("===============================")
    print(f"Total Tasks: {totalTasks}")
    for method in searchMethods:
        if method in accuracies:
            correct=accuracies[method]
            accuracyPerc=(correct/totalTasks)*100
            methodLabel=method.upper()
            if 'custom' in method:
                methodLabel+=f" (Custom Heuristic {method[-1]})"
            print(f"{methodLabel} Accuracy: {correct}/{totalTasks} ({accuracyPerc:.1f}%)")
    print("===============================")


if __name__=="__main__":
    #Test with different search methods
    runTests(searchMethods=['bfs','gbfs','astar','gbfs_custom1','astar_custom1'],maxComplexity=5)

