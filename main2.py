# Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.

# Example 1:

# Input: 
# [
#   [1,1,1],
#   [1,0,1],
#   [1,1,1]
# ]
# Output: 
# [
#   [1,0,1],
#   [0,0,0],
#   [1,0,1]
# ]

class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        isFCZero = False
        
        for i in range(len(matrix)):
            if matrix[i][0] == 0:
#                     print("Setting isFCZero = True because matrix[{}][0] == 0".format(i))
                isFCZero = True
            for j in range(1,len(matrix[0])):
#                 print(i,j)
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
                    
        for i in range(1,len(matrix)):
            for j in range(1,len(matrix[0])):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        
        if matrix[0][0] == 0:
            for i in range(len(matrix[0])):
                matrix[0][i] = 0
                
 
        if isFCZero:
            for i in range(len(matrix)):
                matrix[i][0] = 0
                
# Game of life

# Given a board with m by n cells, each cell has an initial state live (1) or dead (0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):

# Any live cell with fewer than two live neighbors dies, as if caused by under-population.
# Any live cell with two or three live neighbors lives on to the next generation.
# Any live cell with more than three live neighbors dies, as if by over-population..
# Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.

class Solution(object):
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        # Any live cell with fewer than two live neighbors dies, as if caused by under-population.
        # Any live cell with two or three live neighbors lives on to the next generation.
        # Any live cell with more than three live neighbors dies, as if by over-population..
        # Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
        dirs = [[0, -1], [-1, 0], [0, 1], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]]
        mr, mc = len(board), len(board[0])
        # 0 or 2 = dead, dead -> alive
        # 1 or 3 = alive, alive -> dead
        for r in range(mr):
            for c in range(mc):
                alive, dead = 0, 0
                for _dir in dirs:
                    nr, nc = r + _dir[0], c + _dir[1]
                    if nr >=0 and nc >= 0 and nr < mr and nc < mc:
                        if board[nr][nc]%2:
                            alive+=1
                        else:
                            dead += 1
                curr = board[r][c]%2
                if (curr and alive < 2) or (curr and alive > 3):
                    board[r][c] = 3
                elif curr == 0 and alive == 3:
                    board[r][c] = 2
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 2: board[i][j] = 1
                if board[i][j] == 3: board[i][j] = 0
                    
                    
                    
                
        
# A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

# The robot can only move either down or right at any point in time.
#  The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

# How many possible unique paths are there?


# Input: m = 3, n = 2
# Output: 3
# Explanation:
# From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
# 1. Right -> Right -> Down
# 2. Right -> Down -> Right
# 3. Down -> Right -> Right

class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        if not m or not n:
            return 0
        
        dp = [[1 for i in range(m)] for j in range(n)]
        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]


# Now consider if some obstacles are added to the grids. How many unique paths would there be?

# Input:
# [
#   [0,0,0],
#   [0,1,0],
#   [0,0,0]
# ]
# Output: 2
# Explanation:
# There is one obstacle in the middle of the 3x3 grid above.
# There are two ways to reach the bottom-right corner:
# 1. Right -> Right -> Down -> Down
# 2. Down -> Down -> Right -> Right

class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """

        m = len(obstacleGrid)
        n = len(obstacleGrid[0])

        # If the starting cell has an obstacle, then simply return as there would be
        # no paths to the destination.
        if obstacleGrid[0][0] == 1:
            return 0

        # Number of ways of reaching the starting cell = 1.
        obstacleGrid[0][0] = 1

        # Filling the values for the first column
        for i in range(1,m):
            obstacleGrid[i][0] = int(obstacleGrid[i][0] == 0 and obstacleGrid[i-1][0] == 1)

        # Filling the values for the first row        
        for j in range(1, n):
            obstacleGrid[0][j] = int(obstacleGrid[0][j] == 0 and obstacleGrid[0][j-1] == 1)

        # Starting from cell(1,1) fill up the values
        # No. of ways of reaching cell[i][j] = cell[i - 1][j] + cell[i][j - 1]
        # i.e. From above and left.
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j] == 0:
                    obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j-1]
                else:
                    obstacleGrid[i][j] = 0

        # Return value stored in rightmost bottommost cell. That is the destination.            
        return obstacleGrid[m-1][n-1]


# There is a ball in a maze with empty spaces and walls. The ball can go through empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction.

# Given the ball's start position, 
# the destination and the maze, 
# determine whether the ball could stop at the destination.

# The maze is represented by a binary 2D array. 
# 1 means the wall and 0 means the empty space. 
# You may assume that the borders of the maze are all walls. 
# The start and destination coordinates are represented by row and column indexes.

class Solution(object):
    def hasPath(self, maze, start, destination):
        """
        :type maze: List[List[int]]
        :type start: List[int]
        :type destination: List[int]
        :rtype: bool
        """
        mr, mc = len(maze), len(maze[0])
        visited = [[False for _ in range(mc)] for _ in range(mr)]
        visited[start[0]][start[1]] = True
        q = collections.deque()
        q.append([start[0], start[1]])
        dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        while q:
            node = q.popleft()
            if node == destination:
                return True
            for _dir in dirs:
                next_node = node[0] + _dir[0], node[1] + _dir[1]
                while next_node[0] >= 0 and next_node[1] >= 0 and next_node[0] < mr and next_node[1] < mc and maze[next_node[0]][next_node[1]] == 0:
                    next_node = next_node[0]+_dir[0], next_node[1]+_dir[1]
                next_node = next_node[0]-_dir[0], next_node[1]-_dir[1]
                if(not visited[next_node[0]][next_node[1]]):
                    q.append([next_node[0], next_node[1]])
                    visited[next_node[0]][next_node[1]] = True
        return False
                    
# Given a non-empty array of integers, return the k most frequent elements.

# Example 1:

# Input: nums = [1,1,1,2,2,3], k = 2
# Output: [1,2]
# Example 2:

# Input: nums = [1], k = 1
# Output: [1]


class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """ #NlogK time, N space
        c = collections.Counter(nums)
        return heapq.nlargest(k, c, key = lambda z: c[z])
        

# Given a non-empty list of words, return the k most frequent elements.

# Your answer should be sorted by frequency from highest to lowest.
#  If two words have the same frequency, then the word with the lower alphabetical order comes first.

# Example 1:
# Input: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
# Output: ["i", "love"]
# Explanation: "i" and "love" are the two most frequent words.
#     Note that "i" comes before "love" due to a lower alphabetical order.

class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        c = collections.Counter(words)
        return(heapq.nsmallest(k, c, lambda z: (-c[z], z)))


# Shortest Path Visiting All Nodes
# An undirected, connected graph of N nodes (labeled 0, 1, 2, ..., N-1) is given as graph.

# graph.length = N, and j != i is in the list graph[i] exactly once, if and only if nodes i and j are connected.

# Return the length of the shortest path that visits every node. You may start and stop at any node, you may revisit nodes multiple times, and you may reuse edges.

 

# Example 1:

# Input: [[1,2,3],[0],[0],[0]]
# Output: 4
# Explanation: One possible path is [1,0,2,0,3]


class Node(object):
    def __init__(self, nodeId, visitedSoFar):
        self.id = nodeId
        self.journal = visitedSoFar
    
    def __eq__(self, other):
        return self.id == other.id and self.journal == other.journal

    def __repr__(self):
        return "Node({}, {})".format(self.id, bin(self.journal)[2:])
    
    def __hash__(self):
        return hash((self.id, self.journal))
    
    
class Solution(object):
    def shortestPathLength(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: int
        """
        N = len(graph)
        # 1<<i represents nodes visitedSoFar to reach this node
        # when initializing, we don't know the best node to start 
        # our journey around the world with. So we add all
        # nodes to our queue aka travel journal !
        q = collections.deque(Node(i, 1<<i) for i in range(N))
        distanceToThisNode = collections.defaultdict(lambda :N*N)
        for i in range(N): 
            distanceToThisNode[Node(i, 1<<i)] = 0
        
        endJournal = (1 << N) - 1
        # when we have visited all nodes, this is how our journal 
        # aka visitedSoFar at that node would look like.
        
        while(q):
            node = q.popleft()
            
            dist = distanceToThisNode[node]
            
            if(node.journal == endJournal):
                return dist 
            
            neighbouring_cities = graph[node.id]
            
            for city in neighbouring_cities:
                newJournal = node.journal | (1 << city)
                # doing an OR operation with 1<<city essentially adds
                # this city to the journal. aka sets that nodeId to 1
                
                neighbour_node = Node(city, newJournal)
                    
                if dist+1 < distanceToThisNode[neighbour_node]:
                    distanceToThisNode[neighbour_node] = dist+1
                    q.append(neighbour_node)
        return -1


# Dungeon Game
# The demons had captured the princess (P) and imprisoned her in the bottom-right corner of a dungeon. The dungeon consists of M x N rooms laid out in a 2D grid. Our valiant knight (K) was initially positioned in the top-left room and must fight his way through the dungeon to rescue the princess.

# The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately.

# Some of the rooms are guarded by demons, so the knight loses health (negative integers) upon entering these rooms; other rooms are either empty (0's) or contain magic orbs that increase the knight's health (positive integers).

# In order to reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.

class Solution(object):
    def calculateMinimumHP(self, dungeon):
        """
        :type dungeon: List[List[int]]
        :rtype: int
        """
        mr, mc  = len(dungeon), len(dungeon[0])
        dp = [[float("inf") for _ in range(mc+1)] for _ in range(mr+1)]
        dp[mr][mc-1], dp[mr-1][mc] = 0, 0
        for r in range(mr-1, -1, -1):
            for c in range(mc -1, -1, -1):
                #dp[r][c] = max(, 0) can't be negative
                dp[r][c] = max(min(dp[r+1][c], dp[r][c+1]) - dungeon[r][c], 0)
        
        return dp[0][0]+1



# 489. Robot Room Cleaner
# Given a robot cleaner in a room modeled as a grid.

# Each cell in the grid can be empty or blocked.

# The robot cleaner with 4 given APIs can move forward, turn left or turn right. 
# Each turn it made is 90 degrees.

# When it tries to move into a blocked cell, 
# its bumper sensor detects the obstacle and it stays on the current cell.

# """
# This is the robot's control interface.
# You should not implement it, or speculate about its implementation
# """
#class Robot(object):
#    def move(self):
#        """
#        Returns true if the cell in front is open and robot moves into the cell.
#        Returns false if the cell in front is blocked and robot stays in the current cell.
#        :rtype bool
#        """
#
#    def turnLeft(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def turnRight(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def clean(self):
#        """
#        Clean the current cell.
#        :rtype void
#        """

class Solution:
    def cleanRoom(self, robot):
        def dfs(x, y, dx, dy, visited):
            # 1, Clean current
            robot.clean()
            visited.add((x, y))

            # 2, Clean next
            for _ in range(4):
                if (x + dx, y + dy) not in visited and robot.move():
                    dfs(x + dx, y + dy, dx, dy, visited)
                    # 3. Come back to same state
                    robot.turnLeft()
                    robot.turnLeft()
                    robot.move()
                    robot.turnLeft()
                    robot.turnLeft()
                # 4 switch direction
                robot.turnLeft()
                dx, dy = -dy, dx

        dfs(0, 0, 0, 1, set())



# Container with most water
# Input: [1,8,6,2,5,4,8,3,7]
# Output: 49

class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        maxarea = 0
        l, r = 0, len(height)-1
        while(l < r):
            maxarea = max(maxarea, min(height[l], height[r])*(r-l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return maxarea

# Given a string containing only digits, restore it by returning all possible valid IP address combinations.

# Example:

# Input: "25525511135"
# Output: ["255.255.11.135", "255.255.111.35"]

class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        self.res = []
        self.dfs(s, 0, "")
        return self.res
        
    def dfs(self, s, index, recovered):
        if index == 4:
            if s == "":
                self.res.append(recovered[:-1])
            return
        for i in range(1,4):
            if i <= len(s):
                if i == 1:
                    self.dfs(s[i:], index + 1, recovered + s[:i] + ".")
                elif i == 2 and s[0] != 0:
                    self.dfs(s[i:], index + 1, recovered + s[:i] + ".")
                elif i == 3 and s[0] != 0 and int(s[:3]) <= 255:
                    self.dfs(s[i:], index + 1, recovered + s[:i] + ".")
                    
                    
            

# Candidate Sum
# Given a set of candidate numbers (candidates) (without duplicates) and a target number (target), 
# find all unique combinations in candidates where the candidate numbers sums to target.


# Input: candidates = [2,3,6,7], target = 7,
# A solution set is:
# [
#   [7],
#   [2,2,3]
# ]

class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        self.ans = []
        candidates.sort()
        self.dfs(candidates, [], 0, target)
        return self.ans
        
    def dfs(self, candidates, res, start, target):
        if target < 0:
            return
        if target == 0:
            self.ans.append(res)
            return
        for i in range(start, len(candidates)):
            self.dfs(candidates, res + [candidates[i]], i, target - candidates[i])



# Integer to Roman
# Symbol       Value
# I             1
# V             5
# X             10
# L             50
# C             100
# D             500
# M             1000



class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        ans = ""
        numerals = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        for i, v in enumerate(values):
            ans += (num//v)*numerals[i]
            num %= v
        return ans
            