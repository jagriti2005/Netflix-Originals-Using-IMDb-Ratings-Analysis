import React, { useState, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, PieChart, Pie, Cell, AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { TrendingUp, Film, Star, Calendar, BarChart3, Target, TrendingDown, Zap } from 'lucide-react';

const NetflixAnalysisDashboard = () => {
  const [activeChart, setActiveChart] = useState('distribution');

  // Sample Netflix Originals data with IMDb ratings
  const netflixData = [
    { title: "Stranger Things", year: 2016, genre: "Sci-Fi", imdb: 8.7, type: "Series", seasons: 4, runtime: 50, budget: 30 },
    { title: "The Crown", year: 2016, genre: "Drama", imdb: 8.6, type: "Series", seasons: 6, runtime: 60, budget: 100 },
    { title: "Ozark", year: 2017, genre: "Crime", imdb: 8.4, type: "Series", seasons: 4, runtime: 60, budget: 25 },
    { title: "The Witcher", year: 2019, genre: "Fantasy", imdb: 8.2, type: "Series", seasons: 3, runtime: 60, budget: 80 },
    { title: "Bridgerton", year: 2020, genre: "Romance", imdb: 7.3, type: "Series", seasons: 2, runtime: 55, budget: 60 },
    { title: "Money Heist", year: 2017, genre: "Crime", imdb: 8.3, type: "Series", seasons: 5, runtime: 70, budget: 40 },
    { title: "Dark", year: 2017, genre: "Sci-Fi", imdb: 8.8, type: "Series", seasons: 3, runtime: 60, budget: 20 },
    { title: "Orange Is the New Black", year: 2013, genre: "Comedy", imdb: 8.1, type: "Series", seasons: 7, runtime: 60, budget: 50 },
    { title: "House of Cards", year: 2013, genre: "Drama", imdb: 8.6, type: "Series", seasons: 6, runtime: 50, budget: 60 },
    { title: "Narcos", year: 2015, genre: "Crime", imdb: 8.8, type: "Series", seasons: 3, runtime: 50, budget: 35 },
    { title: "Roma", year: 2018, genre: "Drama", imdb: 7.7, type: "Movie", seasons: 0, runtime: 135, budget: 15 },
    { title: "The Irishman", year: 2019, genre: "Crime", imdb: 7.8, type: "Movie", seasons: 0, runtime: 209, budget: 159 },
    { title: "Marriage Story", year: 2019, genre: "Drama", imdb: 7.9, type: "Movie", seasons: 0, runtime: 137, budget: 18 },
    { title: "Bird Box", year: 2018, genre: "Horror", imdb: 6.6, type: "Movie", seasons: 0, runtime: 124, budget: 19 },
    { title: "The Kissing Booth", year: 2018, genre: "Romance", imdb: 5.9, type: "Movie", seasons: 0, runtime: 105, budget: 5 },
    { title: "Extraction", year: 2020, genre: "Action", imdb: 6.7, type: "Movie", seasons: 0, runtime: 116, budget: 65 },
    { title: "The Old Guard", year: 2020, genre: "Action", imdb: 6.6, type: "Movie", seasons: 0, runtime: 125, budget: 70 },
    { title: "Enola Holmes", year: 2020, genre: "Mystery", imdb: 6.6, type: "Movie", seasons: 0, runtime: 123, budget: 25 },
  ];

  // Chart 1: Rating Distribution (Histogram) - Best for showing frequency distribution
  const ratingDistribution = useMemo(() => {
    const bins = [
      { range: "5.0-6.0", count: 0, color: "#ef4444" },
      { range: "6.0-7.0", count: 0, color: "#f97316" },
      { range: "7.0-8.0", count: 0, color: "#eab308" },
      { range: "8.0-9.0", count: 0, color: "#22c55e" },
      { range: "9.0-10.0", count: 0, color: "#16a34a" }
    ];
    
    netflixData.forEach(item => {
      if (item.imdb >= 5.0 && item.imdb < 6.0) bins[0].count++;
      else if (item.imdb >= 6.0 && item.imdb < 7.0) bins[1].count++;
      else if (item.imdb >= 7.0 && item.imdb < 8.0) bins[2].count++;
      else if (item.imdb >= 8.0 && item.imdb < 9.0) bins[3].count++;
      else if (item.imdb >= 9.0) bins[4].count++;
    });
    
    return bins;
  }, []);

  // Chart 2: Genre Performance (Bar Chart) - Best for comparing categories
  const genrePerformance = useMemo(() => {
    const genreStats = {};
    netflixData.forEach(item => {
      if (!genreStats[item.genre]) {
        genreStats[item.genre] = { genre: item.genre, total: 0, count: 0, avgRating: 0 };
      }
      genreStats[item.genre].total += item.imdb;
      genreStats[item.genre].count++;
    });
    
    return Object.values(genreStats)
      .map(g => ({
        ...g,
        avgRating: parseFloat((g.total / g.count).toFixed(1))
      }))
      .sort((a, b) => b.avgRating - a.avgRating);
  }, []);

  // Chart 3: Time Series (Line Chart) - Best for showing trends over time
  const yearlyTrends = useMemo(() => {
    const yearStats = {};
    netflixData.forEach(item => {
      if (!yearStats[item.year]) {
        yearStats[item.year] = { year: item.year, total: 0, count: 0, avgRating: 0 };
      }
      yearStats[item.year].total += item.imdb;
      yearStats[item.year].count++;
    });
    
    return Object.values(yearStats)
      .map(y => ({
        ...y,
        avgRating: parseFloat((y.total / y.count).toFixed(1))
      }))
      .sort((a, b) => a.year - b.year);
  }, []);

  // Chart 4: Correlation (Scatter Plot) - Best for showing relationships
  const budgetRatingCorrelation = netflixData.map(item => ({
    budget: item.budget,
    imdb: item.imdb,
    title: item.title,
    type: item.type
  }));

  // Chart 5: Content Type Distribution (Pie Chart) - Best for showing parts of a whole
  const contentTypeDistribution = useMemo(() => {
    const typeStats = {};
    netflixData.forEach(item => {
      typeStats[item.type] = (typeStats[item.type] || 0) + 1;
    });
    
    return Object.entries(typeStats).map(([type, count]) => ({
      name: type,
      value: count,
      percentage: ((count / netflixData.length) * 100).toFixed(1)
    }));
  }, []);

  // Chart 6: Rating vs Runtime (Area Chart) - Best for showing distribution over continuous variable
  const runtimeRatingData = netflixData
    .filter(item => item.type === 'Movie')
    .sort((a, b) => a.runtime - b.runtime)
    .map(item => ({
      runtime: item.runtime,
      imdb: item.imdb,
      title: item.title
    }));

  const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1'];

  const chartConfigs = [
    {
      id: 'distribution',
      title: 'IMDb Rating Distribution',
      subtitle: 'Histogram - Best for showing frequency distribution of ratings',
      icon: <BarChart3 className="w-5 h-5" />,
      reasoning: 'Histogram is ideal for showing how ratings are distributed across different ranges, revealing the overall quality pattern of Netflix Originals.'
    },
    {
      id: 'genre',
      title: 'Average Rating by Genre',
      subtitle: 'Bar Chart - Best for comparing performance across categories',
      icon: <Film className="w-5 h-5" />,
      reasoning: 'Bar charts excel at comparing values across different categories, making it easy to identify which genres perform best.'
    },
    {
      id: 'trends',
      title: 'Rating Trends Over Time',
      subtitle: 'Line Chart - Best for showing temporal trends',
      icon: <TrendingUp className="w-5 h-5" />,
      reasoning: 'Line charts are perfect for showing how average ratings have changed over time, revealing quality trends.'
    },
    {
      id: 'correlation',
      title: 'Budget vs IMDb Rating',
      subtitle: 'Scatter Plot - Best for showing relationships between variables',
      icon: <Zap className="w-5 h-5" />,
      reasoning: 'Scatter plots reveal correlations between two continuous variables, helping identify if higher budgets lead to better ratings.'
    },
    {
      id: 'content-type',
      title: 'Content Type Distribution',
      subtitle: 'Pie Chart - Best for showing parts of a whole',
      icon: <Target className="w-5 h-5" />,
      reasoning: 'Pie charts effectively show proportional relationships, revealing the balance between series and movies.'
    },
    {
      id: 'runtime',
      title: 'Movie Runtime vs Rating',
      subtitle: 'Area Chart - Best for showing continuous distribution',
      icon: <Calendar className="w-5 h-5" />,
      reasoning: 'Area charts work well for showing how ratings vary across a continuous variable like runtime.'
    }
  ];

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-semibold">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const renderChart = () => {
    switch (activeChart) {
      case 'distribution':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={ratingDistribution} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="range" />
              <YAxis />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'genre':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={genrePerformance} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="genre" angle={-45} textAnchor="end" height={80} />
              <YAxis domain={[0, 10]} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="avgRating" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'trends':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={yearlyTrends} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis domain={[0, 10]} />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="avgRating" stroke="#8884d8" strokeWidth={3} dot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'correlation':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="budget" name="Budget (Million $)" />
              <YAxis dataKey="imdb" name="IMDb Rating" domain={[5, 9]} />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                        <p className="font-semibold">{data.title}</p>
                        <p>Budget: ${data.budget}M</p>
                        <p>IMDb: {data.imdb}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter data={budgetRatingCorrelation} fill="#8884d8" />
            </ScatterChart>
          </ResponsiveContainer>
        );

      case 'content-type':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie
                data={contentTypeDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percentage }) => `${name} (${percentage}%)`}
                outerRadius={120}
                fill="#8884d8"
                dataKey="value"
              >
                {contentTypeDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        );

      case 'runtime':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={runtimeRatingData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="runtime" />
              <YAxis domain={[5, 9]} />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                        <p className="font-semibold">{data.title}</p>
                        <p>Runtime: {data.runtime} min</p>
                        <p>IMDb: {data.imdb}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Area type="monotone" dataKey="imdb" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
            </AreaChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  const currentConfig = chartConfigs.find(config => config.id === activeChart);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Netflix Originals Analysis
          </h1>
          <p className="text-xl text-gray-600">
            Selecting Appropriate Chart Types for Data Insights
          </p>
        </div>

        {/* Chart Selection */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          {chartConfigs.map((config) => (
            <button
              key={config.id}
              onClick={() => setActiveChart(config.id)}
              className={`p-4 rounded-lg border-2 transition-all duration-200 text-left ${
                activeChart === config.id
                  ? 'border-blue-500 bg-blue-50 shadow-lg'
                  : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-md'
              }`}
            >
              <div className="flex items-center gap-3 mb-2">
                {config.icon}
                <h3 className="font-semibold text-gray-900">{config.title}</h3>
              </div>
              <p className="text-sm text-gray-600 mb-2">{config.subtitle}</p>
            </button>
          ))}
        </div>

        {/* Current Chart Display */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            {currentConfig?.icon}
            <h2 className="text-2xl font-bold text-gray-900">{currentConfig?.title}</h2>
          </div>
          <p className="text-gray-600 mb-6">{currentConfig?.subtitle}</p>
          
          {renderChart()}
        </div>

        {/* Chart Reasoning */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-3">Why This Chart Type?</h3>
          <p className="text-gray-700 leading-relaxed">
            {currentConfig?.reasoning}
          </p>
          
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <h4 className="font-semibold text-blue-900 mb-2">Key Insights from Current Chart:</h4>
            <ul className="text-blue-800 space-y-1">
              {activeChart === 'distribution' && (
                <>
                  <li>• Most Netflix Originals score between 7.0-8.0 on IMDb</li>
                  <li>• Very few shows/movies score below 6.0 or above 9.0</li>
                  <li>• The distribution shows Netflix maintains consistent quality</li>
                </>
              )}
              {activeChart === 'genre' && (
                <>
                  <li>• Crime and Sci-Fi genres have the highest average ratings</li>
                  <li>• Romance and Horror genres tend to score lower</li>
                  <li>• Drama maintains consistently high ratings across titles</li>
                </>
              )}
              {activeChart === 'trends' && (
                <>
                  <li>• Netflix's content quality has remained relatively stable over time</li>
                  <li>• Early originals (2013-2017) set a high standard</li>
                  <li>• Recent years show slight fluctuations but maintain quality</li>
                </>
              )}
              {activeChart === 'correlation' && (
                <>
                  <li>• Higher budgets don't always guarantee better ratings</li>
                  <li>• Some low-budget productions achieve excellent ratings</li>
                  <li>• Content quality depends more on execution than budget</li>
                </>
              )}
              {activeChart === 'content-type' && (
                <>
                  <li>• Series make up the majority of Netflix Originals</li>
                  <li>• Movies represent a smaller but significant portion</li>
                  <li>• Series format allows for deeper storytelling and character development</li>
                </>
              )}
              {activeChart === 'runtime' && (
                <>
                  <li>• Movie runtime doesn't strongly correlate with rating</li>
                  <li>• Both short and long movies can achieve high ratings</li>
                  <li>• Story quality matters more than duration</li>
                </>
              )}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NetflixAnalysisDashboard;