# HitCounter

class HitCounter(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        from collections import deque
        
        self.num_of_hits = 0
        self.time_hits = deque()
        

    def hit(self, timestamp):
        """
        Record a hit.
        @param timestamp - The current timestamp (in seconds granularity).
        :type timestamp: int
        :rtype: void
        """
        if not self.time_hits or self.time_hits[-1][0] != timestamp:
            self.time_hits.append([timestamp, 1])
        else:
            self.time_hits[-1][1] += 1
        
        self.num_of_hits += 1
                
        

    def getHits(self, timestamp):
        """
        Return the number of hits in the past 5 minutes.
        @param timestamp - The current timestamp (in seconds granularity).
        :type timestamp: int
        :rtype: int
        """
        while self.time_hits and self.time_hits[0][0] <= timestamp - 300:
            self.num_of_hits -= self.time_hits.popleft()[1]
        
        return self.num_of_hits



# Boundary of a binary Tree

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def traverseLeft(self, root):
        while(root):
            if root == None:
                return
            if root.left:
                self.leftB.append(root.val)
                root = root.left
            elif root.right:
                self.leftB.append(root.val)
                root = root.right
            else:
                return
            # else:
            #     self.leftB.append(root.val)
            #     return
    
    def traverseRight(self, root):
        while(root):
            if root == None:
                return
            if root.right:
                self.rightB.append(root.val)
                root = root.right
            elif root.left:
                self.rightB.append(root.val)
                root = root.left
            else:
                return
    
    def getLeaves(self, node):
        if node and not node.left and not node.right:
            self.leaves.append(node.val)
        if node and node.left:
            self.getLeaves(node.left)
        if node and node.right:
            self.getLeaves(node.right)
            
    
    def boundaryOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        self.leftB, self.rightB, self.leaves, res = [], [], [], []
        if not root:
            return []
        if root.left:
            self.traverseLeft(root.left)
        if root.right:
            self.traverseRight(root.right)
        self.getLeaves(root)
        # print( self.leftB, self.leaves, self.rightB[::-1])
        if root.left or root.right:
            res.append(root.val)
        res += self.leftB + self.leaves + self.rightB[::-1]
        
        return(res)


# Reorder Log Files
# Input: ["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"]
# Output: ["g1 act car","a8 act zoo","ab1 off key dog","a1 9 2 3 1","zo4 4 7"]
# You have an array of logs.  Each log is a space delimited string of words.

# For each log, the first word in each log is an alphanumeric identifier.  Then, either:

# Each word after the identifier will consist only of lowercase letters, or;
# Each word after the identifier will consist only of digits.
# We will call these two varieties of logs letter-logs and digit-logs.  It is guaranteed that each log has at least one word after its identifier.

# Reorder the logs so that all of the letter-logs come before any digit-log.  The letter-logs are ordered lexicographically ignoring identifier, with the identifier used in case of ties.  The digit-logs should be put in their original order.

# Return the final order of the logs.

class Solution(object):
    def reorderLogFiles(self, logs):
        """
        :type logs: List[str]
        :rtype: List[str]
        """
        def f(log):
            lid, rest = log.split(" ", 1)
            return (0, rest, lid) if rest[-1].isalpha() else (1,)
        return sorted(logs, key = f)
        

# Binary Tree Zigzag Level Order Traversal
# Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).


class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        def levelOrder(root, levelList, currLevel):
            if not root:
                return levelList
            if root not in levelList[currLevel]:
                levelList[currLevel].append(root.val)
                if root.left: levelList = levelOrder(root.left, levelList, currLevel+1)
                if root.right: levelList = levelOrder(root.right, levelList, currLevel+1)
            return levelList
        levelList = levelOrder(root, collections.defaultdict(list), 0)
        res = []
        for idx in range(len(levelList)):
            if idx%2 != 0:
                levelList[idx].reverse()
            res.append(levelList[idx])
        return res



# Peeking Iterator


# Given an Iterator class interface with methods: next() and hasNext(), 
# design and implement a PeekingIterator that support the peek() operation -- 
# it essentially peek() at the element that will be returned by the next call to next().

# Below is the interface for Iterator, which is already defined for you.
#
# class Iterator(object):
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """

class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iterator = iterator
        self._has_next = iterator.hasNext()
        self.peeked_val = None
        self.peeked = False

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        if not self.peeked:
            self.peeked_val = self.iterator.next()
            self.peeked = True
        return self.peeked_val

    def next(self):
        """
        :rtype: int
        """
        if self.hasNext:
            val = self.peek()
            #next aligns
            self.peeked = False
            self._has_next = self.iterator.hasNext()
            return val
        
    def hasNext(self):
        return self._has_next
        

# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()         # Should return the same value as [val].



class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        def fdfs(image, r, c, newColor):
            if image[r][c] == oldColor:
                image[r][c] = newColor
                
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if nr >= 0 and nc >= 0 and nr < mr and nc < mc and image[nr][nc] == oldColor:
                        fdfs(image, nr, nc, newColor)
            return
                        
        
        oldColor = image[sr][sc]
        if oldColor == newColor:
            return image
        mr, mc = len(image), len(image[0])   
        dirs = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        fdfs(image, sr, sc, newColor)
        return image


# Given an array of meeting time intervals consisting of start and end times 
# [[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend all meetings.

class Solution(object):
    def canAttendMeetings(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: bool
        """
        if len(intervals) < 1:
            return True
        intervals.sort(key = lambda k: k.start)
        last_int = intervals[0]
        for interval in intervals[1:]:
            if interval.start < last_int.end:
                return False
            last_int = interval
        return True


# Given an array of meeting time intervals consisting of start and end times 
# [[s1,e1],[s2,e2],...] (si < ei), 
# find the minimum number of conference rooms required.

class Solution(object):
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: int
        """
        starts = sorted([interval.start for interval in intervals])
        ends = sorted([interval.end for interval in intervals])
        s, e = 0, 0
        used_rooms = 0
        while s < len(intervals):
            # If there is a meeting that has ended by the time the meeting at `start_pointer` starts
            if(starts[s] >= ends[e]):
                used_rooms -= 1
                e += 1
            used_rooms += 1
            s += 1
        return used_rooms

# There are a total of n courses you have to take, labeled from 0 to n-1.

# Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

# Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

# Example 1:

# Input: 2, [[1,0]] 
# Output: true
# Explanation: There are a total of 2 courses to take. 
#              To take course 1 you should have finished course 0. So it is possible.


class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        in_degree, out_degree, zero_degree = collections.defaultdict(set), collections.defaultdict(set), collections.deque()
        
        for i, j in prerequisites:
            # [1, 0]   0 j----> 1 i
            in_degree[i].add(j)
            out_degree[j].add(i)
        
        for i in range(numCourses):
            if i not in in_degree:
                zero_degree.append(i)
        
        while zero_degree:
            prereq = zero_degree.popleft()
            for course in out_degree[prereq]:
                in_degree[course].discard(prereq)
                if not in_degree[course]:
                    # we can now take this course, lets break the chain
                    zero_degree.append(course)
            del out_degree[prereq]
        
        if out_degree:
            return False
        return True


# There are a total of n courses you have to take, labeled from 0 to n-1.

# Some courses may have prerequisites, 
# for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

# Given the total number of courses and a list of prerequisite pairs,
#  return the ordering of courses you should take to finish all courses.

# There may be multiple correct orders, you just need to return one of them. 
# If it is impossible to finish all courses, return an empty array.

from collections import defaultdict
class Solution:

    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """

        # Prepare the graph
        adj_list = defaultdict(list)
        indegree = {}
        for dest, src in prerequisites:
            adj_list[src].append(dest)

            # Record each node's in-degree
            indegree[dest] = indegree.get(dest, 0) + 1

        # Queue for maintainig list of nodes that have 0 in-degree
        zero_indegree_queue = [k for k in range(numCourses) if k not in indegree]

        topological_sorted_order = []

        # Until there are nodes in the Q
        while zero_indegree_queue:

            # Pop one node with 0 in-degree
            vertex = zero_indegree_queue.pop(0)
            topological_sorted_order.append(vertex)

            # Reduce in-degree for all the neighbors
            if vertex in adj_list:
                for neighbor in adj_list[vertex]:
                    indegree[neighbor] -= 1

                    # Add neighbor to Q if in-degree becomes 0
                    if indegree[neighbor] == 0:
                        zero_indegree_queue.append(neighbor)

        return topological_sorted_order if len(topological_sorted_order) == numCourses else []

# Input: [[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]
# Output: 3
# Explanation: 
# There're totally 4 courses, but you can take 3 courses at most:
# First, take the 1st course, it costs 100 days so you will finish it on the 100th day, and ready to take the next course on the 101st day.
# Second, take the 3rd course, it costs 1000 days so you will finish it on the 1100th day, and ready to take the next course on the 1101st day. 
# Third, take the 2nd course, it costs 200 days so you will finish it on the 1300th day. 
# The 4th course cannot be taken now, since you will finish it on the 3300th day, which exceeds the closed date.

# public class Solution {
#     public int scheduleCourse(int[][] courses) {
#         Arrays.sort(courses, (a, b) -> a[1] - b[1]);
#         PriorityQueue < Integer > queue = new PriorityQueue < > ((a, b) -> b - a);
#         int time = 0;
#         for (int[] c: courses) {
#             if (time + c[0] <= c[1]) {
#                 queue.offer(c[0]);
#                 time += c[0];
#             } else if (!queue.isEmpty() && queue.peek() > c[0]) {
#                 time += c[0] - queue.poll();
#                 queue.offer(c[0]);
#             }
#         }
#         return queue.size();
#     }
# }

# Given a string S and a string T, find the minimum window in S 
# which will contain all the characters in T in complexity O(n).

# Example:

# Input: S = "ADOBECODEBANC", T = "ABC"
# Output: "BANC"

class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        window_dict = collections.defaultdict(int)
        t_dict = collections.Counter(t)
        l, r = 0, 0
        ans = (float("inf"), l, r)
        
        formed, reqd = 0, len(t_dict)
        
        while r < len(s):
            # print(l, r, window_dict)
            character = s[r]
            
            window_dict[character] += 1
            
            if character in t_dict and window_dict[character] == t_dict[character]:
                formed += 1
            # print(l, r, window_dict, formed, reqd)
            while l <= r and formed == reqd:
                # print(l, r, window_dict, formed, reqd)
                character = s[l]
                
                if(r-l+1 < ans[0]):
                    ans = (r-l+1, l, r)
                        
                window_dict[character] -= 1
                
                if character in t_dict and window_dict[character] < t_dict[character]:
                    formed -= 1
                        
                        
                l += 1
            r += 1
        return "" if ans[0] == float("inf") else s[ans[1]:ans[2]+1]
            

# Serialize and Deserialize BST

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if root == None:
            return "None"
        return str(root.val) + "," + self.serialize(root.left) + "," + self.serialize(root.right)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        nodes = data.split(",")
        def _deserialize(nodelist):
            if nodelist[0] == "None":
                nodelist.pop(0)
                return None
            root = TreeNode(nodelist.pop(0))
            # this is the key here - pop after creating node
            root.left = _deserialize(nodelist)
            root.right = _deserialize(nodelist)
            return root
        t = _deserialize(nodes)
        return t
        
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))


# Serialize and Deserialize Binary Tree

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if root == None:
            return "None"
        return str(root.val) + "," + self.serialize(root.left) + "," + self.serialize(root.right)
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        def dfs(data_list):
            if data_list[0] == "None":
                data_list.pop(0)
                return None
            t1 = TreeNode(data_list.pop(0))
            t1.left = dfs(data_list)
            t1.right = dfs(data_list)
            return t1
        return dfs(data.split(","))
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))


#  Integer to English Words
# Input: 123
# Output: "One Hundred Twenty Three"


class Solution(object):
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        def tens(num):
            if not num:
                return ""
            elif num < 10:
                return ones[num]
            elif num < 20:
                return tens_one[num]
            else:
                tenner = num //10
                rest = num - tenner*10
                #when calculating rest no need to divide
                return tens_norm[tenner] + " " + ones[rest] if rest else tens_norm[tenner]
            
            
        def toWords(num):
            # print(num)
            hundred = num // 100
            rest = num - hundred * 100
            #when calculating rest no need to divide
            if hundred and rest:
                return ones[hundred] + " Hundred " + tens(rest)
            elif hundred and not rest:
                return ones[hundred] + " Hundred"
            else:
                return tens(rest)
                
        ones = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
        tens_norm = {2: "Twenty", 3: "Thirty", 4: "Forty", 5: "Fifty", 6: "Sixty", 7: "Seventy", 8: "Eighty", 9: "Ninety"}
        tens_one = {10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen", 16: "Sixteen", 17: "Seventeen", 18: "Eighteen", 19: "Nineteen"}
        #split
        
        billion = num//(10**9)
        million = (num - billion*10**9)//(10**6)
        thousand = (num  - billion * 10**9 - million * 10**6)//1000
        rest = num - billion*(10**9) - (million*10**6) - (thousand*1000)
        
        if not num:
            return "Zero"
        
        result = ""
        if billion:
            result += toWords(billion) + " Billion"
        if million:
            result += " " if result else ""
            result += toWords(million) + " Million"
        if thousand:
            result += " " if result else ""
            result += toWords(thousand) + " Thousand"
        if rest:
            result += " " if result else ""
            result += toWords(rest)
        return result
            
        
# Search in Rotated Sorted Array

# Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

# (i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        def find_pivot(left, right):
            # 4 5 6 7 8 9 2
            # the nums[0] is always  
            # greater than nums[-1] 
            # in case of pivoted array
            if nums[left] < nums[right]:
                return 0
            while left <= right:
                pivot = (left + right) // 2
                if nums[pivot] > nums[pivot+1]:
                    # we're basically finding 
                    # an anomaly where nums[i] > nums[i+1]
                    return pivot +1
                else:
                    if nums[pivot] < nums[left]:
                        # restrict search to left side
                        # aka put right to p-1
                        right = pivot -1
                    else:
                        left = pivot +1
        def search(left, right):
            while left<= right:
                pivot = (left + right)//2
                if nums[pivot] == target:
                    return pivot
                elif nums[pivot] < target:
                    left = pivot + 1
                else:
                    right = pivot - 1
            return -1
        
        N = len(nums)
        if N == 0:
            return -1
        if N == 1:
            return 0 if nums[0] == target else -1
        
        pivot = find_pivot(0, len(nums)-1)
        if nums[pivot] == target:
            return pivot
        
        if pivot == 0:
            return search(0, N-1)
        if target < nums[0]:
            return search(pivot, N-1)
        return search(0, pivot)
            
                    
# Given a matrix of M x N elements (M rows, N columns), 
# return all elements of the matrix in diagonal order as shown in the below image. 

class Solution(object):
    def findDiagonalOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        result = [ ]
        dd = collections.defaultdict(list)
        if not matrix: return result
        # Step 1: Numbers are grouped by the diagonals.
        # Numbers in same diagonal have same value of row+col
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                dd[i+j+1].append(matrix[i][j]) # starting indices from 1, hence i+j+1.
        # Step 2: Place diagonals in the result list.
        # But remember to reverse numbers in odd diagonals.
        for k in sorted(dd.keys()):
            if k%2==1: dd[k].reverse()
            result += dd[k]
        return result


#  Binary Tree Maximum Path Sum
# Given a non-empty binary tree, find the maximum path sum.

# For this problem, a path is defined as any sequence of nodes from some 
# starting node to any node in the tree along the parent-child connections. 
# The path must contain at least one node and does not need to go through the root. 

class Solution:
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def max_gain(node):
            nonlocal max_sum
            if not node:
                return 0

            # max sum on the left and right sub-trees of node
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)
            
            # the price to start a new path where `node` is a highest node
            price_newpath = node.val + left_gain + right_gain
            
            # update max_sum if it's better to start a new path
            max_sum = max(max_sum, price_newpath)
        
            # for recursion :
            # return the max gain if continue the same path
            return node.val + max(left_gain, right_gain)
   
        max_sum = float('-inf')
        max_gain(root)
        return max_sum



        