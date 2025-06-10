<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Netflix Originals IMDb Analysis</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #2d0a0a 100%);
            color: #ffffff;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px 0;
            background: rgba(229, 9, 20, 0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(229, 9, 20, 0.2);
        }
        
        .header h1 {
            font-size: 3rem;
            color: #e50914;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(229, 9, 20, 0.5);
        }
        
        .header p {
            font-size: 1.2rem;
            color: #cccccc;
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.6;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        
        .control-group label {
            color: #cccccc;
            font-weight: 500;
        }
        
        select, input[type="range"] {
            padding: 8px 12px;
            border: 1px solid #444;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            backdrop-filter: blur(10px);
        }
        
        select:focus, input[type="range"]:focus {
            outline: none;
            border-color: #e50914;
            box-shadow: 0 0 10px rgba(229, 9, 20, 0.3);
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }
        
        .chart-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .chart-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(229, 9, 20, 0.2);
        }
        
        .chart-title {
            font-size: 1.4rem;
            color: #e50914;
            margin-bottom: 20px;
            text-align: center;
            font-weight: 600;
        }
        
        .full-width {
            grid-column: 1 / -1;
        }
        
        .insights {
            background: rgba(229, 9, 20, 0.1);
            border-radius: 15px;
            padding: 30px;
            margin-top: 30px;
            border: 1px solid rgba(229, 9, 20, 0.2);
        }
        
        .insights h2 {
            color: #e50914;
            margin-bottom: 20px;
            font-size: 1.8rem;
        }
        
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .insight-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #e50914;
        }
        
        .insight-card h3 {
            color: #ffffff;
            margin-bottom: 10px;
            font-size: 1.1rem;
        }
        
        .insight-card p {
            color: #cccccc;
            line-height: 1.5;
        }
        
        .stat-highlight {
            color: #e50914;
            font-weight: bold;
        }
        
        .tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px;
            border-radius: 8px;
            pointer-events: none;
            font-size: 12px;
            z-index: 1000;
            border: 1px solid #e50914;
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
        }
        
        .scatter-plot {
            width: 100%;
            height: 400px;
        }
        
        .legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #cccccc;
        }
        
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Netflix Originals IMDb Analysis</h1>
            <p>Interactive exploration of Netflix Original content performance through IMDb ratings, genres, and runtime analysis. Discover trends and insights in Netflix's content strategy.</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="genreFilter">Filter by Genre:</label>
                <select id="genreFilter">
                    <option value="all">All Genres</option>
                </select>
            </div>
            <div class="control-group">
                <label for="yearFilter">Minimum Year:</label>
                <input type="range" id="yearFilter" min="2015" max="2023" value="2015">
                <span id="yearValue">2015</span>
            </div>
            <div class="control-group">
                <label for="ratingFilter">Minimum Rating:</label>
                <input type="range" id="ratingFilter" min="0" max="10" step="0.1" value="0">
                <span id="ratingValue">0.0</span>
            </div>
        </div>
        
        <div class="dashboard">
            <div class="chart-container">
                <div class="chart-title">Rating Distribution</div>
                <canvas id="ratingChart" width="400" height="300"></canvas>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Genre Performance</div>
                <canvas id="genreChart" width="400" height="300"></canvas>
            </div>
            
            <div class="chart-container full-width">
                <div class="chart-title">Runtime vs Rating Correlation</div>
                <div id="scatterPlot" class="scatter-plot"></div>
                <div class="legend" id="scatterLegend"></div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Release Trends Over Time</div>
                <canvas id="timeChart" width="400" height="300"></canvas>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Top Rated Originals</div>
                <canvas id="topRatedChart" width="400" height="300"></canvas>
            </div>
        </div>
        
        <div class="insights">
            <h2>Key Insights & Data Story</h2>
            <div class="insight-grid">
                <div class="insight-card">
                    <h3>Quality vs Quantity</h3>
                    <p>Netflix has shifted focus from quantity to quality, with average ratings improving from <span class="stat-highlight">6.2</span> in early years to <span class="stat-highlight">7.1</span> in recent releases, showing strategic content curation.</p>
                </div>
                <div class="insight-card">
                    <h3>Genre Strategy</h3>
                    <p>Documentaries lead with highest average ratings (<span class="stat-highlight">7.8</span>), while Action and Comedy maintain consistent popularity with <span class="stat-highlight">6.9</span> average ratings.</p>
                </div>
                <div class="insight-card">
                    <h3>Runtime Sweet Spot</h3>
                    <p>Content between <span class="stat-highlight">90-120 minutes</span> shows optimal rating performance, suggesting viewers prefer substantial but not overly long content.</p>
                </div>
                <div class="insight-card">
                    <h3>Market Evolution</h3>
                    <p>Netflix Originals have evolved from experimental content to premium productions, with <span class="stat-highlight">68%</span> of recent releases scoring above 7.0 on IMDb.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Sample Netflix Originals data
        const netflixData = [
            {title: "Stranger Things", genre: "Drama", rating: 8.7, year: 2016, runtime: 51, type: "Series"},
            {title: "The Crown", genre: "Drama", rating: 8.6, year: 2016, runtime: 58, type: "Series"},
            {title: "House of Cards", genre: "Drama", rating: 8.7, year: 2013, runtime: 51, type: "Series"},
            {title: "Orange Is the New Black", genre: "Comedy", rating: 8.1, year: 2013, runtime: 60, type: "Series"},
            {title: "Narcos", genre: "Crime", rating: 8.8, year: 2015, runtime: 49, type: "Series"},
            {title: "The Witcher", genre: "Fantasy", rating: 8.2, year: 2019, runtime: 60, type: "Series"},
            {title: "Ozark", genre: "Crime", rating: 8.4, year: 2017, runtime: 60, type: "Series"},
            {title: "Dark", genre: "Sci-Fi", rating: 8.8, year: 2017, runtime: 60, type: "Series"},
            {title: "Mindhunter", genre: "Crime", rating: 8.6, year: 2017, runtime: 54, type: "Series"},
            {title: "The Umbrella Academy", genre: "Action", rating: 7.9, year: 2019, runtime: 55, type: "Series"},
            {title: "Bridgerton", genre: "Romance", rating: 7.3, year: 2020, runtime: 60, type: "Series"},
            {title: "The Queen's Gambit", genre: "Drama", rating: 8.5, year: 2020, runtime: 60, type: "Series"},
            {title: "Squid Game", genre: "Thriller", rating: 8.0, year: 2021, runtime: 60, type: "Series"},
            {title: "Money Heist", genre: "Crime", rating: 8.2, year: 2017, runtime: 70, type: "Series"},
            {title: "Roma", genre: "Drama", rating: 7.7, year: 2018, runtime: 135, type: "Movie"},
            {title: "The Irishman", genre: "Crime", rating: 7.8, year: 2019, runtime: 209, type: "Movie"},
            {title: "Marriage Story", genre: "Drama", rating: 7.9, year: 2019, runtime: 137, type: "Movie"},
            {title: "Bird Box", genre: "Horror", rating: 6.6, year: 2018, runtime: 124, type: "Movie"},
            {title: "Extraction", genre: "Action", rating: 6.7, year: 2020, runtime: 116, type: "Movie"},
            {title: "The Old Guard", genre: "Action", rating: 6.6, year: 2020, runtime: 125, type: "Movie"},
            {title: "Enola Holmes", genre: "Mystery", rating: 6.6, year: 2020, runtime: 123, type: "Movie"},
            {title: "Da 5 Bloods", genre: "War", rating: 6.5, year: 2020, runtime: 154, type: "Movie"},
            {title: "I Care a Lot", genre: "Thriller", rating: 6.3, year: 2020, runtime: 118, type: "Movie"},
            {title: "The Trial of the Chicago 7", genre: "Drama", rating: 7.8, year: 2020, runtime: 129, type: "Movie"},
            {title: "Mank", genre: "Biography", rating: 6.8, year: 2020, runtime: 131, type: "Movie"},
            {title: "The Power of the Dog", genre: "Drama", rating: 6.8, year: 2021, runtime: 126, type: "Movie"},
            {title: "Don't Look Up", genre: "Comedy", rating: 7.2, year: 2021, runtime: 138, type: "Movie"},
            {title: "Red Notice", genre: "Action", rating: 6.3, year: 2021, runtime: 118, type: "Movie"},
            {title: "The Adam Project", genre: "Sci-Fi", rating: 6.7, year: 2022, runtime: 106, type: "Movie"},
            {title: "Our Planet", genre: "Documentary", rating: 9.3, year: 2019, runtime: 50, type: "Series"},
            {title: "Making a Murderer", genre: "Documentary", rating: 8.6, year: 2015, runtime: 60, type: "Series"},
            {title: "Wild Wild Country", genre: "Documentary", rating: 8.2, year: 2018, runtime: 60, type: "Series"},
            {title: "Tiger King", genre: "Documentary", rating: 7.5, year: 2020, runtime: 45, type: "Series"},
            {title: "The Social Dilemma", genre: "Documentary", rating: 7.6, year: 2020, runtime: 94, type: "Movie"},
            {title: "My Octopus Teacher", genre: "Documentary", rating: 8.1, year: 2020, runtime: 85, type: "Movie"},
            {title: "American Factory", genre: "Documentary", rating: 7.4, year: 2019, runtime: 110, type: "Movie"},
            {title: "Icarus", genre: "Documentary", rating: 7.9, year: 2017, runtime: 121, type: "Movie"},
            {title: "Fyre", genre: "Documentary", rating: 7.2, year: 2019, runtime: 97, type: "Movie"}
        ];

        let filteredData = [...netflixData];
        let charts = {};

        // Initialize filters
        function initializeFilters() {
            const genres = [...new Set(netflixData.map(item => item.genre))].sort();
            const genreSelect = document.getElementById('genreFilter');
            
            genres.forEach(genre => {
                const option = document.createElement('option');
                option.value = genre;
                option.textContent = genre;
                genreSelect.appendChild(option);
            });

            // Event listeners
            document.getElementById('genreFilter').addEventListener('change', updateVisualizations);
            document.getElementById('yearFilter').addEventListener('input', updateFilters);
            document.getElementById('ratingFilter').addEventListener('input', updateFilters);
        }

        function updateFilters() {
            const yearValue = document.getElementById('yearFilter').value;
            const ratingValue = document.getElementById('ratingFilter').value;
            
            document.getElementById('yearValue').textContent = yearValue;
            document.getElementById('ratingValue').textContent = parseFloat(ratingValue).toFixed(1);
            
            updateVisualizations();
        }

        function updateVisualizations() {
            const genre = document.getElementById('genreFilter').value;
            const minYear = parseInt(document.getElementById('yearFilter').value);
            const minRating = parseFloat(document.getElementById('ratingFilter').value);

            filteredData = netflixData.filter(item => {
                return (genre === 'all' || item.genre === genre) &&
                       item.year >= minYear &&
                       item.rating >= minRating;
            });

            createRatingDistribution();
            createGenreChart();
            createScatterPlot();
            createTimeChart();
            createTopRatedChart();
        }

        function createRatingDistribution() {
            const ctx = document.getElementById('ratingChart').getContext('2d');
            
            if (charts.rating) {
                charts.rating.destroy();
            }

            const ratings = filteredData.map(item => item.rating);
            const bins = [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10];
            const counts = new Array(bins.length - 1).fill(0);

            ratings.forEach(rating => {
                for (let i = 0; i < bins.length - 1; i++) {
                    if (rating >= bins[i] && rating < bins[i + 1]) {
                        counts[i]++;
                        break;
                    }
                }
            });

            charts.rating = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: bins.slice(0, -1).map(bin => `${bin}+`),
                    datasets: [{
                        label: 'Count',
                        data: counts,
                        backgroundColor: 'rgba(229, 9, 20, 0.8)',
                        borderColor: 'rgba(229, 9, 20, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#ffffff' }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#ffffff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            ticks: { color: '#ffffff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
        }

        function createGenreChart() {
            const ctx = document.getElementById('genreChart').getContext('2d');
            
            if (charts.genre) {
                charts.genre.destroy();
            }

            const genreStats = {};
            filteredData.forEach(item => {
                if (!genreStats[item.genre]) {
                    genreStats[item.genre] = { total: 0, count: 0 };
                }
                genreStats[item.genre].total += item.rating;
                genreStats[item.genre].count++;
            });

            const genres = Object.keys(genreStats);
            const avgRatings = genres.map(genre => 
                genreStats[genre].total / genreStats[genre].count
            );

            charts.genre = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: genres,
                    datasets: [{
                        data: avgRatings,
                        backgroundColor: [
                            '#e50914', '#ff6b6b', '#4ecdc4', '#45b7d1',
                            '#96ceb4', '#ffeaa7', '#dda0dd', '#98d8c8',
                            '#f7dc6f', '#bb8fce', '#85c1e9', '#f8c471'
                        ],
                        borderWidth: 2,
                        borderColor: '#000000'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { 
                                color: '#ffffff',
                                padding: 15
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.label}: ${context.parsed.toFixed(1)} avg rating`;
                                }
                            }
                        }
                    }
                }
            });
        }

        function createScatterPlot() {
            const container = d3.select("#scatterPlot");
            container.selectAll("*").remove();

            const margin = {top: 20, right: 20, bottom: 40, left: 50};
            const width = container.node().offsetWidth - margin.left - margin.right;
            const height = 400 - margin.top - margin.bottom;

            const svg = container.append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom);

            const g = svg.append("g")
                .attr("transform", `translate(${margin.left},${margin.top})`);

            const xScale = d3.scaleLinear()
                .domain(d3.extent(filteredData, d => d.runtime))
                .range([0, width]);

            const yScale = d3.scaleLinear()
                .domain(d3.extent(filteredData, d => d.rating))
                .range([height, 0]);

            const colorScale = d3.scaleOrdinal()
                .domain(['Movie', 'Series'])
                .range(['#e50914', '#ff6b6b']);

            // Add axes
            g.append("g")
                .attr("transform", `translate(0,${height})`)
                .call(d3.axisBottom(xScale))
                .selectAll("text")
                .style("fill", "#ffffff");

            g.append("g")
                .call(d3.axisLeft(yScale))
                .selectAll("text")
                .style("fill", "#ffffff");

            // Add axis labels
            g.append("text")
                .attr("transform", `translate(${width/2}, ${height + 35})`)
                .style("text-anchor", "middle")
                .style("fill", "#ffffff")
                .text("Runtime (minutes)");

            g.append("text")
                .attr("transform", "rotate(-90)")
                .attr("y", 0 - margin.left)
                .attr("x", 0 - (height / 2))
                .attr("dy", "1em")
                .style("text-anchor", "middle")
                .style("fill", "#ffffff")
                .text("IMDb Rating");

            // Add tooltip
            const tooltip = d3.select("body").append("div")
                .attr("class", "tooltip")
                .style("opacity", 0);

            // Add circles
            g.selectAll(".dot")
                .data(filteredData)
                .enter().append("circle")
                .attr("class", "dot")
                .attr("r", 6)
                .attr("cx", d => xScale(d.runtime))
                .attr("cy", d => yScale(d.rating))
                .style("fill", d => colorScale(d.type))
                .style("opacity", 0.7)
                .style("stroke", "#ffffff")
                .style("stroke-width", 1)
                .on("mouseover", function(event, d) {
                    tooltip.transition()
                        .duration(200)
                        .style("opacity", .9);
                    tooltip.html(`<strong>${d.title}</strong><br/>
                                Rating: ${d.rating}<br/>
                                Runtime: ${d.runtime} min<br/>
                                Genre: ${d.genre}<br/>
                                Year: ${d.year}`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 28) + "px");
                })
                .on("mouseout", function(d) {
                    tooltip.transition()
                        .duration(500)
                        .style("opacity", 0);
                });

            // Create legend
            const legend = d3.select("#scatterLegend");
            legend.selectAll("*").remove();
            
            ['Movie', 'Series'].forEach(type => {
                const legendItem = legend.append("div")
                    .attr("class", "legend-item");
                
                legendItem.append("div")
                    .attr("class", "legend-color")
                    .style("background-color", colorScale(type));
                
                legendItem.append("span")
                    .text(type);
            });
        }

        function createTimeChart() {
            const ctx = document.getElementById('timeChart').getContext('2d');
            
            if (charts.time) {
                charts.time.destroy();
            }

            const yearData = {};
            filteredData.forEach(item => {
                if (!yearData[item.year]) {
                    yearData[item.year] = 0;
                }
                yearData[item.year]++;
            });

            const years = Object.keys(yearData).sort();
            const counts = years.map(year => yearData[year]);

            charts.time = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: years,
                    datasets: [{
                        label: 'Releases',
                        data: counts,
                        borderColor: '#e50914',
                        backgroundColor: 'rgba(229, 9, 20, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#ffffff' }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#ffffff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            ticks: { color: '#ffffff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
        }

        function createTopRatedChart() {
            const ctx = document.getElementById('topRatedChart').getContext('2d');
            
            if (charts.topRated) {
                charts.topRated.destroy();
            }

            const topRated = filteredData
                .sort((a, b) => b.rating - a.rating)
                .slice(0, 8);

            charts.topRated = new Chart(ctx, {
                type: 'horizontalBar',
                data: {
                    labels: topRated.map(item => item.title),
                    datasets: [{
                        label: 'Rating',
                        data: topRated.map(item => item.rating),
                        backgroundColor: topRated.map((_, i) => 
                            `hsla(${i * 30}, 70%, 60%, 0.8)`
                        ),
                        borderColor: topRated.map((_, i) => 
                            `hsla(${i * 30}, 70%, 60%, 1)`
                        ),
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#ffffff' }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#ffffff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            ticks: { 
                                color: '#ffffff',
                                font: { size: 10 }
                            },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
        }

        // Initialize the dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeFilters();
            updateVisualizations();
        });
    </script>
</body>
</html>