from manim import *

class SIGProjectExplanation(Scene):
    def construct(self):
        # 1. Title
        title = Text("Graph Importance Sampling (GIS)").scale(0.9)
        subtitle = Text("Fast Neural Network Pruning").scale(0.7).next_to(title, DOWN)
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        # 2. Building the Graph
        step1 = Text("1. Model to Graph").to_edge(UP)
        self.play(Write(step1))
        
        # Create a simple neural network graph
        layers = [3, 4, 3]
        nodes = VGroup()
        edges = VGroup()
        
        # Create nodes
        for i, num_nodes in enumerate(layers):
            layer_nodes = VGroup(*[Circle(radius=0.2, color=BLUE, fill_opacity=0.5) for _ in range(num_nodes)])
            layer_nodes.arrange(DOWN, buff=0.5)
            layer_nodes.move_to(RIGHT * (i - 1) * 3)
            nodes.add(layer_nodes)
            
        self.play(FadeIn(nodes))
        
        # Create edges
        for i in range(len(layers)-1):
            for n1 in nodes[i]:
                for n2 in nodes[i+1]:
                    edge = Line(n1.get_right(), n2.get_left(), stroke_width=2, stroke_opacity=0.3)
                    edges.add(edge)
                    
        self.play(Create(edges), run_time=2)
        self.wait(1)

        # 3. Computing Centrality
        step2 = Text("2. Compute Eigenvector Centrality").to_edge(UP)
        self.play(Transform(step1, step2))
        
        # Highlight some nodes with high centrality
        highlight = nodes[1][1].copy().set_color(YELLOW).set_fill(YELLOW, opacity=0.8)
        self.play(FadeIn(highlight))
        centrality_label = Text("High Centrality").scale(0.4).next_to(highlight, UP)
        self.play(Write(centrality_label))
        self.wait(1.5)
        
        # 4. Backward Propagation
        step3 = Text("3. Backward Importance Propagation").to_edge(UP)
        self.play(Transform(step1, step3))
        self.play(FadeOut(centrality_label))
        
        # Animate flow backward from the central node
        arrows = VGroup()
        for n1 in nodes[0]:
            arrow = Arrow(highlight.get_left(), n1.get_right(), color=YELLOW, buff=0.1)
            arrows.add(arrow)
            
        self.play(Create(arrows))
        self.wait(1)
        self.play(FadeOut(arrows), FadeOut(highlight))

        # 5. The Bottleneck & Greedy Algorithm
        self.play(FadeOut(nodes), FadeOut(edges), FadeOut(step1))
        
        bottleneck = Text("Bottleneck: Exact Power Iteration takes 50-500 steps").scale(0.6)
        self.play(Write(bottleneck))
        self.wait(1.5)
        
        solution = Text("Our Solution: Greedy GIS").scale(0.8).set_color(GREEN)
        solution.next_to(bottleneck, DOWN, buff=1)
        self.play(Write(solution))
        
        details = Text("Stop early at K=3 iterations\n~30x Speedup with 96% Rank Correlation").scale(0.6).next_to(solution, DOWN)
        self.play(FadeIn(details))
        self.wait(2.5)
        
        self.play(FadeOut(bottleneck), FadeOut(solution), FadeOut(details))

        # 6. Pruning
        step4 = Text("4. Structured Pruning").to_edge(UP)
        self.play(Write(step4))
        
        self.play(FadeIn(nodes), FadeIn(edges))
        
        # Prune unimportant nodes
        crosses = VGroup()
        nodes_to_prune = [nodes[1][0], nodes[1][3], nodes[0][2]]
        
        for n in nodes_to_prune:
            cross = Cross(n, stroke_color=RED, stroke_width=4)
            crosses.add(cross)
            
        self.play(Create(crosses))
        self.wait(1)
        
        # Fade out pruned nodes and their edges
        pruned_elements = VGroup(*crosses, *nodes_to_prune)
        self.play(FadeOut(pruned_elements))
        
        final_text = Text("Result: Smaller, Faster Model").scale(0.8).to_edge(DOWN)
        self.play(Write(final_text))
        self.wait(2.5)
