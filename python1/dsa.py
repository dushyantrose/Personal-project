# Linklist Code


class Node:

  def __init__(self,value):
    self.data = value # value or data 
    self.next = None # empty Node address of node 
     

a = Node(1) # value store in node
b = Node(2) # value store in node
c = Node(3) # value store in node
     

print(c.data) # print data or value
     
# 3 0utput 

a.next = b  #connect the node
b.next = c  #connect the node
     

print(c.next) # address of node
     
# None(output)

# int(0x7fde268dfdd0) address of node
     
# 140592106307024  address of node

class Node:

  def __init__(self,value):
    self.data = value
    self.next = None
     

class LinkedList:

  def __init__(self):

    # Empty Linked List code of  Empty Linked
    self.head = None
    # no of nodes in the LL
    self.n = 0
 # lenth of linklist 
  def __len__(self):
    return self.n

  def insert_head(self,value): # insert new node

    # new node
    new_node = Node(value)

    # create connection
    new_node.next = self.head

    # reassign head
    self.head = new_node

    # increment n
    self.n = self.n + 1
    # L  = LinkedList()
    # L.insert_head(1)
    # L.insert_head(2)

  def traverse(self) # travrse the element 
    curr = self.head # start from starting point
    while curr ! = None: # until last element
        print(curr.data)
        curr = curr.next # increment by 1
     # or
        
  def __str__(self):

    curr = self.head

    result = ''

    while curr != None:
      result = result + str(curr.data) + '->'
      curr = curr.next

    return result[:-2] # for slicing

  def append(self,value): # insert from tail

    new_node = Node(value) # create new node
   # if list is empty
    if self.head == None:
      # 
      self.head = new_node
      self.n = self.n + 1
      return

    curr = self.head

    while curr.next != None:
      curr = curr.next

    # you are at the last node
    curr.next = new_node
    self.n = self.n + 1

  def insert_after(self,after,value):

    new_node = Node(value)

    curr = self.head

    while curr != None:
      if curr.data == after:
        break
      curr = curr.next

    if curr != None:
      new_node.next = curr.next
      curr.next = new_node
      self.n = self.n + 1
    else:
      return 'Item not found'

  def clear(self):
    self.head = None
    self.n = 0

  def delete_head(self):

    if self.head == None:
      # empty
      return 'Empty LL'

    self.head = self.head.next
      self.n = self.n - 1

  def pop(self):

    if self.head == None:
      # empty
      return 'Empty LL'

    curr = self.head

    # kya linked list me 1 item hai?
    if curr.next == None:
      # head hi hoga(delete from head)
      return self.delete_head()
      

    while curr.next.next != None:
      curr = curr.next

    # curr -> 2nd last node
    curr.next = None
    self.n = self.n - 1

  def remove(self,value):

    if self.head == None:
      return 'Empty LL'

    if self.head.data == value:
      # you want to remove the head node
      return self.delete_head()

    curr = self.head

    while curr.next != None:
      if curr.next.data == value:
        break
      curr = curr.next

    # 2 cases item mil gaya
    # item nai mila
    if curr.next == None:
      # item nai mila
      return 'Not Found'
    else:
      curr.next = curr.next.next
      self.n = self.n - 1

  def search(self,item):

    curr = self.head
    pos = 0

    while curr != None:
      if curr.data == item:
        return pos
      curr = curr.next
      pos = pos + 1

    return 'Not Found'

  def __getitem__(self,index):

    curr = self.head
    pos = 0

    while curr != None:
      if pos == index:
        return curr.data
      curr = curr.next
      pos = pos + 1

    return 'IndexError'


    

    

    
     

L = LinkedList()
     

L.insert_head(1)
L.append(2)
L.append(3)
L.delete_head()
L.insert_head(4)
     

print(L)
     
#  4->4->1->2->3->2->3 output 

L[4]
     
#'IndexError'

print(L)