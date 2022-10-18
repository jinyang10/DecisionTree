import java.io.Serializable;
import java.util.ArrayList;
import java.text.*;
import java.lang.Math;
import java.util.Stack;

public class DecisionTree implements Serializable {

	DTNode rootDTNode;
	int minSizeDatalist; //minimum number of datapoints that should be present in the dataset so as to initiate a split
	
	// Mention the serialVersionUID explicitly in order to avoid getting errors while deserializing.
	public static final long serialVersionUID = 343L;
	
	public DecisionTree(ArrayList<Datum> datalist , int min) {
		minSizeDatalist = min;
		rootDTNode = (new DTNode()).fillDTNode(datalist);
	}

	class DTNode implements Serializable{
		//Mention the serialVersionUID explicitly in order to avoid getting errors while deserializing.
		public static final long serialVersionUID = 438L;
		boolean leaf;
		int label = -1;      // only defined if node is a leaf
		int attribute; // only defined if node is not a leaf, holds the INDEX of the attribute that split is made on
		double threshold;  // only defined if node is not a leaf

		DTNode left, right; //the left and right child of a particular node. (null if leaf)

		DTNode() {
			leaf = true;
			threshold = Double.MAX_VALUE; // threshold = the value at which the split is made
		}

		//returns True if all datums have the same label, false otherwise
		private boolean checkLabels(ArrayList<Datum> datalist) {

			int label = datalist.get(0).y;

			for (int i=0; i<datalist.size(); i++) {

				if (i==0) {
					continue;
				}
				int curLabel = datalist.get(i).y;

				if (label != curLabel) {
					return false;
				}
			}
			return true;
		}

		//calculate avg entropy of 2 datasets
		private double getAvgEntropy(ArrayList<Datum> datalist1, ArrayList<Datum> datalist2) {

			double weight1 = (double) datalist1.size() / ( (double) datalist1.size() + (double) datalist2.size());
			double weight2 = (double) datalist2.size() / ( (double) datalist1.size() + (double) datalist2.size());

			return weight1 * calcEntropy(datalist1) + weight2 * calcEntropy(datalist2);
		}

		// this method takes in a datalist (ArrayList of type datum). It returns the calling DTNode object 
		// as the root of a decision tree trained using the datapoints present in the datalist variable and minSizeDatalist.
		// Also, KEEP IN MIND that the left and right child of the node correspond to "less than" and "greater than or equal to" threshold
		DTNode fillDTNode(ArrayList<Datum> datalist) {

			if (datalist.size() >= minSizeDatalist) {

				//if all data points have same label, return a leaf node with that label
				if (checkLabels(datalist)) {
					DTNode node = new DTNode();
					node.label = datalist.get(0).y;
					return node;

				} else {

					//define a good split
					//go thru each datum, calculate entropy based on an attribute value - choose best entropy
					double bestAvgEntropy = Double.MAX_VALUE;
					int bestAttr = -1;
					double threshold = -1.0;

					int numAttributes = datalist.get(0).x.length;

					for (int attr = 0; attr<numAttributes; attr++) {
						for (Datum point : datalist) {
							double attrValue = point.x[attr];

							ArrayList<Datum> subset1 = new ArrayList<>();
							ArrayList<Datum> subset2 = new ArrayList<>();

							//split data based on chosen attribute
							for (Datum p : datalist) {
								if (p.x[attr] < attrValue) {
									subset1.add(p);
								} else {
									subset2.add(p);
								}
							}
							double avgEntropy = getAvgEntropy(subset1, subset2);
							if (avgEntropy < bestAvgEntropy) {
								bestAvgEntropy = avgEntropy;
								bestAttr = attr;
								threshold = attrValue;
							}
						}
					}

					//if minimum avg entropy found is equal to entropy of input data set,
					//no point in performing a split; node should be a leaf w the majority label
					if (bestAvgEntropy == calcEntropy(datalist)) {
						DTNode node = new DTNode();
						node.label = findMajority(datalist);
						return node;
					}

					//create new node and store attribute test
					DTNode newNode = new DTNode();
					newNode.leaf = false;
					newNode.attribute = bestAttr;
					newNode.threshold = threshold;

					//split dataset based on the attribute
					ArrayList<Datum> subset1 = new ArrayList<>();
					ArrayList<Datum> subset2 = new ArrayList<>();

					for (Datum point : datalist) {
						if (point.x[bestAttr] < threshold) {
							subset1.add(point);
						} else {
							subset2.add(point);
						}
					}
					newNode.left = fillDTNode(subset1);
					newNode.right = fillDTNode(subset2);
					return newNode;
				}

			} else {
				DTNode node = new DTNode();
				int label = findMajority(datalist);
				node.label = label;
				return node;
			}
			
		}



		// helper method. Given a datalist, this method returns the label that has the most
		// occurrences. In case of a tie it returns the label with the smallest value (numerically) involved in the tie.
		int findMajority(ArrayList<Datum> datalist) {
			
			int [] votes = new int[2];

			//loop through the data and count the occurrences of datapoints of each label
			for (Datum data : datalist)
			{
				votes[data.y]+=1;
			}
			
			if (votes[0] >= votes[1])
				return 0;
			else
				return 1;
		}




		// This method takes in a datapoint (excluding the label) in the form of an array of type double (Datum.x) and
		// returns its corresponding label, as determined by the decision tree
		int classifyAtNode(double[] xQuery) {
			if (this.leaf) {
				return this.label;

			} else {
				if (xQuery[this.attribute] < this.threshold) {
					return this.left.classifyAtNode(xQuery);
				} else {
					return this.right.classifyAtNode(xQuery);
				}

			}
		}

		//given another DTNode object, this method checks if the tree rooted at the calling DTNode is equal to the tree rooted
		//at DTNode object passed as the parameter
		public boolean equals(Object dt2)
		{
			if (this == dt2) return true;
			if (dt2 == null || getClass() != dt2.getClass()) return false;

			DTNode node = (DTNode) dt2;
			Stack<DTNode> dt2Stack = new Stack<>();
			Stack<DTNode> thisStack = new Stack<>();

			dt2Stack.push(node);
			thisStack.push(this);

			//traverse each tree, compare nodes as we go
			while (!dt2Stack.isEmpty() && !thisStack.isEmpty()) {

				DTNode dt2Cur = dt2Stack.pop();
				DTNode thisCur = thisStack.pop();

				//one's a leaf and the other's an internal node
				if (dt2Cur.leaf != thisCur.leaf) {
					return false;

					//both are either leafs or internal nodes
				} else {
					if (dt2Cur.leaf) {
						if (dt2Cur.label != thisCur.label) {
							return false;
						}
					} else {
						if (dt2Cur.attribute != thisCur.attribute || dt2Cur.threshold != thisCur.threshold) {
							return false;
						}
					}
				}

				if (dt2Cur.left != null) {
					dt2Stack.push(dt2Cur.left);
				}
				if (dt2Cur.right != null) {
					dt2Stack.push(dt2Cur.right);
				}
				if (thisCur.left != null) {
					thisStack.push(thisCur.left);
				}
				if (thisCur.right != null) {
					thisStack.push(thisCur.right);
				}
			}

			//either finished traversing each tree, or the 2 trees have different lengths
			return dt2Stack.isEmpty() == thisStack.isEmpty();

		}
	}



	//Given a dataset, this returns the entropy of the dataset
	double calcEntropy(ArrayList<Datum> datalist) {
		double entropy = 0;
		double px = 0;
		float [] counter= new float[2];
		if (datalist.size()==0)
			return 0;
		double num0 = 0.00000001,num1 = 0.000000001;

		//calculates the number of points belonging to each of the labels
		for (Datum d : datalist)
		{
			counter[d.y]+=1;
		}
		//calculates the entropy using the formula specified in the document
		for (int i = 0 ; i< counter.length ; i++)
		{
			if (counter[i]>0)
			{
				px = counter[i]/datalist.size();
				entropy -= (px*Math.log(px)/Math.log(2));
			}
		}

		return entropy;
	}


	// given a datapoint (without the label) calls the DTNode.classifyAtNode() on the rootnode of the calling DecisionTree object
	int classify(double[] xQuery ) {
		return this.rootDTNode.classifyAtNode( xQuery );
	}

	// Checks the performance of a DecisionTree on a dataset
	// This method is provided in case you would like to compare your
	// results with the reference values provided in the PDF in the Data
	// section of the PDF
	String checkPerformance( ArrayList<Datum> datalist) {
		DecimalFormat df = new DecimalFormat("0.000");
		float total = datalist.size();
		float count = 0;

		for (int s = 0 ; s < datalist.size() ; s++) {
			double[] x = datalist.get(s).x;
			int result = datalist.get(s).y;
			if (classify(x) != result) {
				count = count + 1;
			}
		}

		return df.format((count/total));
	}


	//Given two DecisionTree objects, this method checks if both the trees are equal by
	//calling onto the DTNode.equals() method
	public static boolean equals(DecisionTree dt1,  DecisionTree dt2)
	{
		boolean flag = true;
		flag = dt1.rootDTNode.equals(dt2.rootDTNode);
		return flag;
	}

}
